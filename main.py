import multiprocessing

import cv2
import torch
import pyvirtualcam
import numpy as np
import mediapipe as mp
from PIL import Image

import tha2.poser.modes.mode_20_wx
from models import TalkingAnimeLight, TalkingAnime3
from pose import get_pose
from utils import preprocessing_image, postprocessing_image

import errno
import json
import os
import queue
import socket
import time
import math
from pynput.mouse import Button, Controller
import re
from collections import OrderedDict
from multiprocessing import Value, Process, Queue

from pyanime4k import ac

from tha2.mocap.ifacialmocap_constants import *

from args import args

from tha3.util import torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image

import collections

# 引入模块
# from ctypes import *
import speech_recognition as sr  # 用16行Python代码实现实时语音识别,用于实时语音识别的主要程序
# 一个计算量和体积都很小的嵌入式语音识别引擎
# from pocketsphinx import LiveSpeech, AudioFile, get_model_path, get_data_path
import wave  # 用于读取wav格式文件
import pyaudio
# import matplotlib.pyplot as plt  # 用于绘制波形图
# import winsound as ws  # winsound 为python内置标准库，可以播放wav格式音频
import contextlib
import threading
from threading import Thread  # Python3中多线程使用的是threading模块。
import random
import pypinyin


def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)


class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0])
        else:
            return 0.0


device = torch.device('cuda') if torch.cuda.is_available() and not args.skip_model else torch.device('cpu')


def create_default_blender_data():
    data = {}

    for blendshape_name in BLENDSHAPE_NAMES:
        data[blendshape_name] = 0.0

    data[HEAD_BONE_X] = 0.0
    data[HEAD_BONE_Y] = 0.0
    data[HEAD_BONE_Z] = 0.0
    data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[LEFT_EYE_BONE_X] = 0.0
    data[LEFT_EYE_BONE_Y] = 0.0
    data[LEFT_EYE_BONE_Z] = 0.0
    data[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[RIGHT_EYE_BONE_X] = 0.0
    data[RIGHT_EYE_BONE_Y] = 0.0
    data[RIGHT_EYE_BONE_Z] = 0.0
    data[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    return data


ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()


class IFMClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = args.ifm.split(':')[0]
        self.port = int(args.ifm.split(':')[1])
        self.ifm_fps_number = Value('f', 0.0)
        self.perf_time = 0

    def run(self):

        udpClntSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        data = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719"

        data = data.encode('utf-8')

        udpClntSock.sendto(data, (self.address, self.port))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind(("", self.port))
        self.socket.settimeout(0.1)
        ifm_fps = FPS()
        pre_socket_string = ''
        while True:
            if self.should_terminate.value:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK or err == 'timed out':
                    continue
                else:
                    raise e
            socket_string = socket_bytes.decode("utf-8")
            if args.debug and pre_socket_string != socket_string:
                self.ifm_fps_number.value = ifm_fps()
                pre_socket_string = socket_string
            # print(socket_string)
            # blender_data = json.loads(socket_string)
            data = self.convert_from_blender_data(socket_string)

            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass
        self.queue.close()
        self.socket.close()

    @staticmethod
    def convert_from_blender_data(blender_data):
        data = {}

        for item in blender_data.split('|'):
            if item.find('#') != -1:
                k, arr = item.split('#')
                arr = [float(n) for n in arr.split(',')]
                data[k.replace("_L", "Left").replace("_R", "Right")] = arr
            elif item.find('-') != -1:
                k, v = item.split("-")
                data[k.replace("_L", "Left").replace("_R", "Right")] = float(v) / 100

        to_rad = 57.3
        data[HEAD_BONE_X] = data["=head"][0] / to_rad
        data[HEAD_BONE_Y] = data["=head"][1] / to_rad
        data[HEAD_BONE_Z] = data["=head"][2] / to_rad
        data[HEAD_BONE_QUAT] = [data["=head"][3], data["=head"][4], data["=head"][5], 1]
        # print(data[HEAD_BONE_QUAT][2],min(data[EYE_BLINK_LEFT],data[EYE_BLINK_RIGHT]))
        data[RIGHT_EYE_BONE_X] = data["rightEye"][0] / to_rad
        data[RIGHT_EYE_BONE_Y] = data["rightEye"][1] / to_rad
        data[RIGHT_EYE_BONE_Z] = data["rightEye"][2] / to_rad
        data[LEFT_EYE_BONE_X] = data["leftEye"][0] / to_rad
        data[LEFT_EYE_BONE_Y] = data["leftEye"][1] / to_rad
        data[LEFT_EYE_BONE_Z] = data["leftEye"][2] / to_rad

        return data


class MouseClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def run(self):
        mouse = Controller()
        posLimit = [int(x) for x in args.mouse_input.split(',')]
        prev = {
            'eye_l_h_temp': 0,
            'eye_r_h_temp': 0,
            'mouth_ratio': 0,
            'eye_y_ratio': 0,
            'eye_x_ratio': 0,
            'x_angle': 0,
            'y_angle': 0,
            'z_angle': 0,
        }
        while True:
            pos = mouse.position
            # print(pos)
            eye_limit = [0.8, 0.5]
            head_eye_reduce = 0.6
            head_slowness = 0.2
            mouse_data = {
                'eye_l_h_temp': 0,
                'eye_r_h_temp': 0,
                'mouth_ratio': 0,
                'eye_y_ratio': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]) * eye_limit[1],
                'eye_x_ratio': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]) * eye_limit[0],
                'x_angle': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]),
                'y_angle': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]),
                'z_angle': 0,
            }
            mouse_data['x_angle'] = np.interp(head_slowness, [0, 1], [prev['x_angle'], mouse_data['x_angle']])
            mouse_data['y_angle'] = np.interp(head_slowness, [0, 1], [prev['y_angle'], mouse_data['y_angle']])
            mouse_data['eye_y_ratio'] -= mouse_data['x_angle'] * eye_limit[1] * head_eye_reduce
            mouse_data['eye_x_ratio'] -= mouse_data['y_angle'] * eye_limit[0] * head_eye_reduce
            if args.bongo:
                mouse_data['y_angle'] += 0.05
                mouse_data['x_angle'] += 0.05
            prev = mouse_data
            self.queue.put_nowait(mouse_data)
            time.sleep(1 / 60)  # 循环程序暂停1/60s


class AIClientProcess(Process):  # leojnjn20221229 创建AIClientProcess对象
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def run(self):
        mouse = Controller()
        posLimit = [int(x) for x in args.AI_input.split(',')]
        prev = {
            'eye_l_h_temp': 0,
            'eye_r_h_temp': 0,
            'mouth_ratio': 0,
            'eye_y_ratio': 0,
            'eye_x_ratio': 0,
            'x_angle': 0,
            'y_angle': 0,
            'z_angle': 0,
        }
        while True:
            pos = mouse.position
            # print(pos)
            eye_limit = [1.0, 0.7]
            head_eye_reduce = 0.6
            head_slowness = 0.2
            mouse_data = {
                'eye_l_h_temp': 0,
                'eye_r_h_temp': 0,
                'mouth_ratio': 0,
                'eye_y_ratio': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]) * eye_limit[1],
                'eye_x_ratio': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]) * eye_limit[0],
                'x_angle': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]),
                'y_angle': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]),
                'z_angle': 0,
            }
            mouse_data['x_angle'] = np.interp(head_slowness, [0, 1], [prev['x_angle'], mouse_data['x_angle']])
            mouse_data['y_angle'] = np.interp(head_slowness, [0, 1], [prev['y_angle'], mouse_data['y_angle']])
            mouse_data['eye_y_ratio'] -= mouse_data['x_angle'] * eye_limit[1] * head_eye_reduce
            mouse_data['eye_x_ratio'] -= mouse_data['y_angle'] * eye_limit[0] * head_eye_reduce
            if args.bongo:
                mouse_data['y_angle'] += 0.05
                mouse_data['x_angle'] += 0.05
            prev = mouse_data
            self.queue.put_nowait(mouse_data)
            time.sleep(1 / 60)  # 循环程序暂停1/60s


class ModelClientProcess(Process):
    def __init__(self, input_image):
        super().__init__()
        self.should_terminate = Value('b', False)
        self.updated = Value('b', False)
        self.data = None
        self.input_image = input_image
        self.output_queue = Queue()
        self.input_queue = Queue()
        self.model_fps_number = Value('f', 0.0)
        self.gpu_fps_number = Value('f', 0.0)
        self.cache_hit_ratio = Value('f', 0.0)
        self.gpu_cache_hit_ratio = Value('f', 0.0)

    def run(self):
        model = None
        if not args.skip_model:
            model = TalkingAnime3().to(device)
            model = model.eval()
            model = model
            print("Pretrained Model Loaded")

        eyebrow_vector = torch.empty(1, 12, dtype=torch.half if args.model.endswith('half') else torch.float)
        mouth_eye_vector = torch.empty(1, 27, dtype=torch.half if args.model.endswith('half') else torch.float)
        pose_vector = torch.empty(1, 6, dtype=torch.half if args.model.endswith('half') else torch.float)

        input_image = self.input_image.to(device)
        eyebrow_vector = eyebrow_vector.to(device)
        mouth_eye_vector = mouth_eye_vector.to(device)
        pose_vector = pose_vector.to(device)

        model_cache = OrderedDict()
        tot = 0
        hit = 0
        hit_in_a_row = 0
        model_fps = FPS()
        gpu_fps = FPS()
        while True:
            model_input = None
            try:
                while not self.input_queue.empty():
                    model_input = self.input_queue.get_nowait()
            except queue.Empty:
                continue
            if model_input is None: continue
            simplify_arr = [1000] * ifm_converter.pose_size
            if args.simplify >= 1:
                simplify_arr = [200] * ifm_converter.pose_size
                simplify_arr[ifm_converter.eye_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_surprised_left_index] = 30
                simplify_arr[ifm_converter.eye_surprised_right_index] = 30
                simplify_arr[ifm_converter.iris_rotation_x_index] = 25
                simplify_arr[ifm_converter.iris_rotation_y_index] = 25
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 10
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 10
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 5
            if args.simplify >= 2:
                simplify_arr[ifm_converter.head_x_index] = 100
                simplify_arr[ifm_converter.head_y_index] = 100
                simplify_arr[ifm_converter.eye_surprised_left_index] = 10
                simplify_arr[ifm_converter.eye_surprised_right_index] = 10
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_left_index]
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_right_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_wink_right_index] / 2
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_right_index] / 2

                uosum = model_input[ifm_converter.mouth_uuu_index] + \
                        model_input[ifm_converter.mouth_ooo_index]
                model_input[ifm_converter.mouth_ooo_index] = uosum
                model_input[ifm_converter.mouth_uuu_index] = 0
                is_open = (model_input[ifm_converter.mouth_aaa_index] + model_input[
                    ifm_converter.mouth_iii_index] + uosum) > 0
                model_input[ifm_converter.mouth_lowered_corner_left_index] = 0
                model_input[ifm_converter.mouth_lowered_corner_right_index] = 0
                model_input[ifm_converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
                model_input[ifm_converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 0
            if args.simplify >= 3:
                simplify_arr[ifm_converter.iris_rotation_x_index] = 20
                simplify_arr[ifm_converter.iris_rotation_y_index] = 20
                simplify_arr[ifm_converter.eye_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_wink_right_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 32
            if args.simplify >= 4:
                simplify_arr[ifm_converter.head_x_index] = 50
                simplify_arr[ifm_converter.head_y_index] = 50
                simplify_arr[ifm_converter.neck_z_index] = 100
                model_input[ifm_converter.eye_raised_lower_eyelid_left_index] = 0
                model_input[ifm_converter.eye_raised_lower_eyelid_right_index] = 0
                simplify_arr[ifm_converter.iris_rotation_x_index] = 10
                simplify_arr[ifm_converter.iris_rotation_y_index] = 10
                simplify_arr[ifm_converter.eye_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_surprised_left_index] = 8
                simplify_arr[ifm_converter.eye_surprised_right_index] = 8
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_wink_right_index]
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2

                model_input[ifm_converter.eye_surprised_left_index] += model_input[
                    ifm_converter.eye_surprised_right_index]
                model_input[ifm_converter.eye_surprised_right_index] = model_input[
                                                                           ifm_converter.eye_surprised_left_index] / 2
                model_input[ifm_converter.eye_surprised_left_index] = model_input[
                                                                          ifm_converter.eye_surprised_left_index] / 2

                model_input[ifm_converter.eye_happy_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.mouth_aaa_index] = min(
                    model_input[ifm_converter.mouth_aaa_index] +
                    model_input[ifm_converter.mouth_ooo_index] / 2 +
                    model_input[ifm_converter.mouth_iii_index] / 2 +
                    model_input[ifm_converter.mouth_uuu_index] / 2, 1
                )
                model_input[ifm_converter.mouth_ooo_index] = 0
                model_input[ifm_converter.mouth_iii_index] = 0
                model_input[ifm_converter.mouth_uuu_index] = 0
            for i in range(4, args.simplify):
                simplify_arr = [max(math.ceil(x * 0.8), 5) for x in simplify_arr]
            for i in range(0, len(simplify_arr)):
                if simplify_arr[i] > 0:
                    model_input[i] = round(model_input[i] * simplify_arr[i]) / simplify_arr[i]
            input_hash = hash(tuple(model_input))
            cached = model_cache.get(input_hash)
            tot += 1
            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            if cached is not None and hit_in_a_row < self.model_fps_number.value:
                self.output_queue.put_nowait(cached)
                model_cache.move_to_end(input_hash)
                hit += 1
                hit_in_a_row += 1
            else:
                hit_in_a_row = 0
                if args.perf == 'model':
                    tic = time.perf_counter()
                if args.eyebrow:
                    for i in range(12):
                        eyebrow_vector[0, i] = model_input[i]
                        eyebrow_vector_c[i] = model_input[i]
                for i in range(27):
                    mouth_eye_vector[0, i] = model_input[i + 12]
                    mouth_eye_vector_c[i] = model_input[i + 12]
                for i in range(6):
                    pose_vector[0, i] = model_input[i + 27 + 12]
                if model is None:
                    output_image = input_image
                else:
                    output_image = model(input_image, mouth_eye_vector, pose_vector, eyebrow_vector, mouth_eye_vector_c,
                                         eyebrow_vector_c,
                                         self.gpu_cache_hit_ratio)
                if args.perf == 'model':
                    torch.cuda.synchronize()
                    print("model", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()
                postprocessed_image = output_image[0].float()
                if args.perf == 'model':
                    print("cpu()", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()
                postprocessed_image = convert_linear_to_srgb((postprocessed_image + 1.0) / 2.0)
                c, h, w = postprocessed_image.shape
                postprocessed_image = 255.0 * torch.transpose(postprocessed_image.reshape(c, h * w), 0, 1).reshape(h, w,
                                                                                                                   c)
                postprocessed_image = postprocessed_image.byte().detach().cpu().numpy()
                if args.perf == 'model':
                    print("postprocess", (time.perf_counter() - tic) * 1000)
                    tic = time.perf_counter()

                self.output_queue.put_nowait(postprocessed_image)
                if args.debug:
                    self.gpu_fps_number.value = gpu_fps()
                if args.max_cache_len > 0:
                    model_cache[input_hash] = postprocessed_image
                    if len(model_cache) > args.max_cache_len:
                        model_cache.popitem(last=False)
            if args.debug:
                self.model_fps_number.value = model_fps()
                self.cache_hit_ratio.value = hit / tot


# if len(phrase_queue) < 3:
#     phrase_queue.append(audio1)
#     phrase_queue.append(audio1)
#     phrase_queue.append(audio1)
# else:
#     phrase_queue.pop(0)
#     phrase_queue.append(audio1)
# 进程方式Process
# ret_volume_queue = multiprocessing.Array('d', range(0), lock=True)
# ret_phrase_queue = multiprocessing.Array('c', [1], lock=True)
# mouth_cues_index = multiprocessing.Value('i', 0, lock=True)
# frame = multiprocessing.Value('i', 0, lock=True)
# volume = multiprocessing.Value('d', 0.0, lock=True)
# np_volume = multiprocessing.Value('d', 0.0, lock=True)
# 线程方式Thread
ret_volume_queue = []  # 使用全局变量的队列，来保存返回值
ret_phrase_queue = ['']
mouth_cues_index = 0  # 嘴型下标
frame = 0
volume = 0.0
np_volume = 0.0


class Wav2MouthThread(Thread):  # class Wav2MouthProcess(Process):  #
    def __init__(self):
        # super().__init__()    #进程方式
        Thread.__init__(self)  # 必须步骤

    def run(self):
        """leojnjn20221206^
        本地音频文件路径
        """
        # here!!!
        # waves_path = rf"data/waves/"  # rf"data/waves/{argsWave}.wav"
        # wave_name = os.listdir(waves_path)  # 得到waves_path文件夹下的所有文件名称
        # wave_path = waves_path + wave_name[5]  # 得到文件夹下音源路径 0 1 2 -> 14 3-> 22 4->98 5->99
        # wave_path = rf"data/waves/{args.wave}.wav"
        wave_path = rf"Rhubarb-Lip-Sync-1.13.0-Windows/waves/{args.wave}.wav"

        # lips_path = rf"data/lips/"
        # lip_name = os.listdir(lips_path)  # 得到lips_path文件夹下的所有文件名称
        # lip_path = lips_path + lip_name[5]  # 得到文件夹下口型json文件路径 0 1 2 -> 14 3-> 22 4->98 5->99
        # lip_path = rf"data/lips/{args.lip}.json"
        lip_path = rf"Rhubarb-Lip-Sync-1.13.0-Windows/lips/{args.lip}.json"

        with open(lip_path, encoding='utf-8') as f:
            data = json.load(f)
        print("lip json Loaded:", args.lip)
        # 计算音频时间
        waveDuration = get_duration_wav(wave_path)
        print(waveDuration)
        duration = data['metadata']['duration']
        print(duration)
        print("-" * 30)
        mouth_cues = data['mouthCues']

        # print("先开始语音到口型计算")
        # # 计算音频时间
        # waveDuration = get_duration_wav(wave_path)
        # print(waveDuration)
        # print("-" * 30)
        # """leojnjn
        # speech_recognition 使用 record()从文件中获取数据,用于实时语音识别的主要程序
        # """
        # recognizer = sr.Recognizer()
        # audio_file = sr.AudioFile(wave_path)
        # # count_time = 0.0
        # # time_gap = 0.5  # 秒0.5s
        # with audio_file as source:
        #     '''默认将文件流的第1秒识别为音频的噪声级别,默认为 1，现将此值下降到 0.1'''
        #     recognizer.adjust_for_ambient_noise(source, duration=0.1)
        #     audio1 = recognizer.record(source, duration=waveDuration)
        #     try:
        #         tik = time.time()
        #         '''Sphinx 不联网，速度快 2.711914300918579s'''
        #         sphinx_text1 = recognizer.recognize_sphinx(audio1, language="zh-CN")  # , language="en-US")  #
        #         # google_text1 = recognizer.recognize_google(audio1, language="cmn-Hans-CN",
        #         #                                            show_all=False)  # , language="en-US")
        #         # ['o', 'u', 'o', 'a', 'e', 'a', 'u', 'e', 'a', 'i', 'e', 'o', 'o', 'i', 'o', 'o', 'e', 'i', 'e']
        #         # vowels = get_vowels(sphinx_text1)  # 英文生搬
        #         vowels = get_vowels_cn(sphinx_text1)  # 拼音准确
        #         tok = time.time()
        #         print(str(tok - tik) + 's')
        #         print("text1 sphinx : {}".format(sphinx_text1))
        #         # print("text1 google : {}".format(google_text1))
        #         # print("pinyin list : {}".format(list_pinyin))
        #         print("vowels: {}".format(vowels))
        #     except sr.UnknownValueError:
        #         print("sphinx could not understand audio")
        #     except sr.RequestError as e:
        #         print("sphinx error; {0}".format(e))
        #     except:
        #         print("Sorry sphinx can't record wave!")

        print("等待15s启图像后")
        time.sleep(15)  # 20用time.sleep模拟任务耗时
        print("开始声音到张嘴o")
        """leojnjn
        读取本地音频波形数据,打开wave文件 file_name[0] = 'lex_000'  words: '您可以制定新的科研方向了司令官'"""
        waveFile = wave.open(wave_path, 'rb')
        print("Wave Audio Loaded:", args.wave)
        # 获取音频的属性参数： nchannels声道数 sampwidth采样大小 framerate帧率 nframes总帧数
        params = waveFile.getparams()
        # 列表：单独提取出各参数的值，并加以定义：
        nchannels, sampwidth, framerate, nframes = params[:4]  # (1, 2, 22050, 89344) #(2, 2, 44100, 5998592)
        print(params[:4])
        waveStr = waveFile.readframes(params[3])  # 读取帧数据params[3]:nframes
        meta = {"seek": 0}  # int变量不能传进函数内部，会有UnboundLocalError，所以给它套上一层壳
        # 读取单通道音频 并绘制波形图（常见音频为左右两个声道）
        # 将字符串转换为16位整数
        waveData = np.frombuffer(waveStr, dtype=np.int16)
        # 归一化：把数据变成（０，１）之间的小数。主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速
        waveData = waveData * 1.0 / (max(abs(waveData)))

        # 要动态生成音频数据或立即处理录制的音频数据，请使用下面概述的回调模式
        # 当需要在执行其他程序时同时播放音频，可以使用回调的方式播放，示例代码如下：每time.sleep(秒)调用
        def callback(in_data, frame_count, time_info, status):
            global frame
            global mouth_cues_index
            global volume
            global np_volume
            vowel = ''  # 元音为何 'a' 'i' 'u' 'e' 'o'

            t = frame / params[2]  # 帧数据params[2]:framerate
            # print(f"current time: {t}")
            '''音量部分：写在这太快太抖了'''
            # if frame < params[3]:  # params[3]:nframes
            #     volume = round(float(abs(waveData[frame])) * 2, 2)  # 使用round内置函数保留2位小数
            #     volume = min(volume, 1.0)  # 限制volume不能超过1
            #     # 张嘴程度0-1
            #     if len(ret_volume_queue) < 3:  # 无长度
            #         ret_volume_queue.append(volume)
            #         ret_volume_queue.append(volume)
            #         ret_volume_queue.append(volume)
            #     else:  # 1长度
            #         ret_volume_queue.pop(0)
            #         ret_volume_queue.append(volume)

            while mouth_cues_index < len(mouth_cues):
                if t < mouth_cues[mouth_cues_index]['start']:
                    break

                # 如果逻辑走到这里，说明要刷新口型了
                index = mouth_cues_index
                # print(mouth_cues[index]['start'])

                '''音量渐变：'''
                t_start = mouth_cues[index]['start']
                t_end = mouth_cues[index]['end']
                t_mid = t_start + (t_end - t_start) / 2.0
                k = 1.0 / (t_end - t_start)

                mouth_shape = mouth_cues[index]['value']
                # print(f"mouth shape: {mouth_shape}")
                '''口型部分：'''
                if "A" == mouth_shape:
                    volume = 0.0  # 限制volume不能超过1
                    np_volume = volume
                else:
                    # volume = 1.0  # 限制volume不能超过1
                    if t >= t_start and t < t_mid:
                        volume = 0.5 + k * (t - t_start)  # 音量渐变
                    elif t >= t_mid and t <= t_end:
                        volume = 1.0 - k * (t - t_mid)  # 音量渐变
                    if "B" == mouth_shape:
                        # volume = 0.5 # 半个正常'e'口型
                        if t >= t_start and t < t_mid:
                            volume = (0.5 + k * (t - t_start)) / 2.0  # 音量渐变
                        elif t >= t_mid and t <= t_end:
                            volume = (1.0 - k * (t - t_mid)) / 2.0  # 音量渐变
                        vowel = 'e'
                    elif "C" == mouth_shape:
                        vowel = 'e'
                    elif "D" == mouth_shape:
                        vowel = 'a'
                    elif "E" == mouth_shape:
                        vowel = 'o'
                    elif "F" == mouth_shape:
                        vowel = 'u'

                    if len(ret_volume_queue) < 3:  # 无长度
                        ret_volume_queue.append(volume)
                        ret_volume_queue.append(volume)
                        ret_volume_queue.append(volume)
                    else:  # 1长度
                        ret_volume_queue.pop(0)
                        ret_volume_queue.append(volume)
                    np_volume = round(np.average(np.array(ret_volume_queue), axis=0, weights=[0.04, 0.16, 0.8]), 2)
                    print(f"ret volume queue: {ret_volume_queue[0]},{ret_volume_queue[1]},{ret_volume_queue[2]}")
                    print(f"np volume: {np_volume}")
                # if "A" != mouth_shape and 0 == volume:
                #     print("*")
                if len(ret_phrase_queue) < 1:  # 无长度
                    ret_phrase_queue.append(vowel)
                else:  # 1长度
                    ret_phrase_queue.pop(0)
                    ret_phrase_queue.append(vowel)

                next_index = index
                while t >= mouth_cues[next_index]['start']:
                    next_index += 1
                    if next_index >= len(mouth_cues):
                        break

                index = next_index - 1
                # print(f"current index: {index}")
                # 这里得到mouth_shape，用于口型表现
                # mouth_shape = mouth_cues[index]['value']
                # # print(f"mouth shape: {mouth_shape}")

                mouth_cues_index = next_index

            frame += frame_count

            # 这是针对流式录制，只有二进制数据没有保存到本地的wav文件时的做法，通过文件指针偏移读取数据
            start = meta["seek"]
            meta["seek"] += frame_count * pyaudio.get_sample_size(pyaudio.paInt16) * waveFile.getnchannels()
            data = waveStr[start: meta["seek"]]
            # 如果有保存成wav文件，直接用文件句柄readframes就行，不用像上面那么麻烦
            # data = waveFile.readframes(frame_count)
            return (data, pyaudio.paContinue)

        pa = pyaudio.PyAudio()
        # 要录制或播放音频，请使用pa.open把音频转化为音频流：上面定义的各个参数，在这里都用上了。
        stream = pa.open(format=pa.get_format_from_width(sampwidth),  # waveFile.getsampwidth()
                         channels=nchannels,  # waveFile.getnchannels()
                         rate=framerate,  # waveFile.getframerate()
                         output=True,
                         stream_callback=callback)  # “回调模式”

        # index = 0
        # count = 0
        # volume = 0
        i_frame = 0
        stream.start_stream()  # read data 使用 pyaudio.Stream.start_stream() 播放/录制音频流
        tic_time = stream.get_time()
        while stream.is_active():  # 直到播放完is_active为False
            # count += 1
            sleep_time = 0.2  # 0.093  # 0.1会有细碎增长 0.2会显得卡
            toc_time = stream.get_time()
            dlt_time = toc_time - tic_time  # 每次会多算的一丢丢点时间
            # print(f'δ流时间：{dlt_time}')
            '''音量部分：写在这还行'''
            if i_frame < nframes:
                volume = round(float(abs(waveData[i_frame])) * 2, 2)  # 使用round内置函数保留2位小数
                i_frame = int(framerate * dlt_time)  # / ratio)  # 4410 # 2205
                volume = min(volume, 1.0)  # 限制volume不能超过1
                if len(ret_volume_queue) < 3:  # 无长度
                    ret_volume_queue.append(volume)
                    ret_volume_queue.append(volume)
                    ret_volume_queue.append(volume)
                else:  # 1长度
                    ret_volume_queue.pop(0)
                    ret_volume_queue.append(volume)
                if 0.0 == volume:
                    np_volume = volume
                    # np_volume = np.average(np.array(ret_volume_queue), axis=0, weights=[0.1, 0.3, 0.6])
                # elif volume < 0.1:  # 默认快张不开嘴用a
                #     ret_phrase_queue[0] = 'a'
                # print(ret_volume_queue[0])  # 打印音量

            #     '''在启用sphinx识别的情况下使用：音量大于0.1且按1秒5元音概算且未超出元音串范围'''
            #     # if ret_volume_queue[0] >= 0.1 and 0 == count % 2 and index < vowels.__len__():  # i_frame % 4410 == 0
            #     #     if len(ret_phrase_queue) < 1:  # 无长度
            #     #         ret_phrase_queue.append(vowels[index])
            #     #     else:  # 1长度
            #     #         ret_phrase_queue.pop(0)
            #     #         ret_phrase_queue.append(vowels[index])
            #     #         index += 1
            #     # else:  # 默认快张不开嘴用a
            #     #     ret_phrase_queue[0] = 'a'
            # print("waiting...")
            time.sleep(sleep_time)  # 0.1
            # print(stream.get_read_available())  # -9977
            # print(stream.get_time())  # 14044.578681822213
        ret_phrase_queue[0] = ''
        ret_volume_queue[0] = 0  # 直到播放完is_active为False,一定得闭嘴
        stream.stop_stream()  # 使用 pyaudio.Stream.stop_stream() 暂停播放/录制
        stream.close()  # 并 pyaudio.Stream.close() 终止流
        waveFile.close()
        pa.terminate()  # 最后，使用 pyaudio.PyAudio.terminate() 终止portaudio会话,关闭播放器

        # # 读取单通道音频 并绘制波形图（常见音频为左右两个声道）
        # # 按照帧数大小的块,读取音频数据：得到一系列二进制编码。
        # waveStr = waveFile.readframes(nframes)
        # # 将字符串转换为16位整数
        # waveData = np.frombuffer(waveStr, dtype=np.int16)
        # # 归一化：把数据变成（０，１）之间的小数。主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速
        # waveData = waveData * 1.0 / (max(abs(waveData)))
        # i_frame = 0
        # while True:
        #     if i_frame < nframes:
        #         volume = int(abs(waveData[i_frame]) * 100)  # 放大取整
        #         print(volume)
        #         i_frame += 2205  # 22050
        #     else:
        #         break
        # i_frame = 0
        # while True:
        #     if i_frame < nframes:
        #         volume = int(abs(waveData[i_frame]) * 100)  # 放大取整
        #         print(volume)
        #         i_frame += 2205  # 22050
        #     else:
        #         break
        # 计算音频时间并绘制
        # waveTime = np.arange(0, nframes) * (1.0 / framerate)  # <class 'numpy.ndarray'>
        # # print(waveTime[2])
        # plt.plot(waveTime, waveData)
        # plt.xlabel("Time(s)")
        # plt.ylabel("Amplitude")
        # plt.title("Single channel wavedata")
        # plt.show()
        testTime = float(nframes / framerate)
        print(testTime)
        print("结束声音到闭嘴-")


class Aud2MouthThread(Thread):
    def __init__(self):
        Thread.__init__(self)  # 必须步骤

    def run(self):
        # print("等待20s先启图像")
        # time.sleep(20)  # 用time.sleep模拟任务耗时
        print("先开始语音到口型")
        """leojnjn20221206^
        本地音频文件路径
        """
        docs_path = rf"data/waves/"  # rf"data/waves/{argsWave}.wav"
        file_name = os.listdir(docs_path)  # 得到文件夹下的所有文件名称
        file_path = docs_path + file_name[0]  # 得到文件夹下音源路径
        # 计算音频时间
        waveDuration = get_duration_wav(file_path)
        print(waveDuration)
        print("-" * 30)
        """leojnjn
        speech_recognition 使用 record()从文件中获取数据,用于实时语音识别的主要程序
        """
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(file_path)
        # count_time = 0.0
        # time_gap = 0.5  # 秒0.5s
        with audio_file as source:
            '''默认将文件流的第1秒识别为音频的噪声级别,默认为 1，现将此值下降到 0.1'''
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            audio1 = recognizer.record(source, duration=waveDuration)
            try:
                tik = time.time()
                '''Sphinx 不联网，速度快 2.711914300918579s'''
                sphinx_text1 = recognizer.recognize_sphinx(audio1, language="en-US")  # , language="zh-CN")  #
                vowels = get_vowels(sphinx_text1)
                tok = time.time()
                print(str(tok - tik) + 's')
                print("text1 sphinx : {}".format(sphinx_text1))
                print("vowels sphinx : {}".format(vowels))
            except sr.UnknownValueError:
                print("sphinx could not understand audio")
            except sr.RequestError as e:
                print("sphinx error; {0}".format(e))
            except:
                print("Sorry sphinx can't record wave!")
        while True:
            if vowels is not None:
                if len(ret_phrase_queue) < 1:  # 无长度
                    ret_phrase_queue.append(vowels[0])
                else:  # 1长度
                    ret_phrase_queue.pop(0)
                    ret_phrase_queue.append(vowels[0])

            # ws.PlaySound(file_path, flags=1)  # winsound可以播放wav格式音频。
            '''0.5秒循环的不要了'''
            # while True:  # waveTime
            #     if count_time < waveDuration:
            #         count_time += time_gap
            #         audio1 = recognizer.record(source, duration=time_gap)
            #         # audio1 = recognizer.record(source, duration=4)
            #         # audio2 = recognizer.record(source, offset=2, duration=2)
            #         # print(type(audio1))
            #         try:
            #             tik = time.time()
            #             '''Sphinx 不联网，速度快 2.711914300918579s'''
            #             sphinx_text1 = recognizer.recognize_sphinx(audio1, language="en-US")  # , language="zh-CN")  #
            #             vowels = get_vowels(sphinx_text1)
            #             tok = time.time()
            #             print(str(tok - tik) + 's')
            #             print("text1 sphinx : {}".format(sphinx_text1))
            #             print("vowels sphinx : {}".format(vowels))
            #         except sr.UnknownValueError:
            #             print("sphinx could not understand audio")
            #         except sr.RequestError as e:
            #             print("sphinx error; {0}".format(e))
            #         except:
            #             print("Sorry sphinx can't record wave!")
            #         if vowels is not None:
            #             if len(ret_phrase_queue) < 1:  # 无长度
            #                 ret_phrase_queue.append(vowels[0])
            #             else:  # 1长度
            #                 ret_phrase_queue.pop(0)
            #                 ret_phrase_queue.append(vowels[0])
            #     else:
            #         break
            # time.sleep(1 / 60)  # 循环程序暂停1/60s

            # try:
            #     tik = time.time()
            #     '''Google Web Speech API 需联网，可选不同的语言识别，速度慢 13.3916015625s
            #     返回了一个关键字为 'alternative' 的列表，指的是全部可能的响应列表'''
            #     google_text1 = recognizer.recognize_google(audio1, language="cmn-Hans-CN",
            #                                                show_all=True)  # , language="en-US")
            #     tok = time.time()
            #     print(str(tok - tik) + 's')
            #     print("text2 google : {}".format(google_text1))
            # except sr.UnknownValueError:
            #     print("google could not understand audio")
            # except sr.RequestError as e:
            #     print("google error; {0}".format(e))
            # except:
            #     print("Sorry google can't record wave!")
            # """leojnjn
            # speech_recognition 用16行Python代码实现实时语音识别,用于实时语音识别的主要程序
            # """
            # listAbleMic = sr.Microphone.list_microphone_names()
            # '''mic = sr.Microphone(device_index=0)但大多数状况下须要使用系统默认麦克风'''
            # mic = sr.Microphone()  # 建立一个Microphone类的实例mic
            # print(listAbleMic)
            # while True:
            #     with mic as source:
            #         recognizer.adjust_for_ambient_noise(source, duration=0.2)
            #         print("Say something please !")
            #         audio_mic = recognizer.listen(source)
            #
            #         try:
            #             text = recognizer.recognize_sphinx(audio_mic, language="zh-CN")  # , language="en-US")
            #             '''Google Web Speech API需联网，可选不同的语言识别，速度慢'''
            #             # text = recognizer.recognize_google(audio)
            #             print("You said : {}".format(text))
            #         except:
            #             print("Sorry I can't hear you!")

            """leojnjn
            pocketsphinx 一个计算量和体积都很小的嵌入式语音识别引擎
            """
            # model_path = get_model_path()
            # print(model_path)
            # data_path = get_data_path()
            # print(data_path)
            # '''
            # 支持的文件格式:wav
            # 音频文件的解码要求: 16KHz, 单声道
            # '''
            # config = {
            #     'verbose': False,
            #     'audio_file': './data/waves/lex_000.wav',
            #     'buffer_size': 2048,
            #     'no_search': False,
            #     'full_utt': False,
            #     # 'hmm': os.path.join(model_path, 'zh_cn.cd_cont_5000'),  # 计算模型
            #     # 'lm': os.path.join(model_path, 'zh_cn.lm.bin'),  # 语言模型
            #     # 'dic': os.path.join(model_path, 'zh_cn.dic')  # 词典模型
            #     'hmm': os.path.join(model_path, 'en-us'),  # 计算模型
            #     'lm': os.path.join(model_path, 'en-us.lm.bin'),  # 语言模型
            #     'dict': os.path.join(model_path, 'cmudict-en-us.dict')  # 词典模型
            # }
            # audio = AudioFile(**config)
            # for phrase in audio:
            #     print("phrase:", phrase)
            #     print(phrase.segments(detailed=True))

            # # 实时录音效果比recognizer.recognize_google(audio)差
            # speech = LiveSpeech(
            #     verbose=False,
            #     sampling_rate=16000,
            #     buffer_size=2048,
            #     no_search=False,
            #     full_utt=False,
            #     # hmm=os.path.join(model_path, 'zh_cn.cd_cont_5000'),
            #     # lm=os.path.join(model_path, 'zh_cn.lm.bin'),
            #     # dic=os.path.join(model_path, 'zh_cn.dic')
            #     hmm=os.path.join(model_path, 'en-us'),
            #     lm=os.path.join(model_path, 'en-us.lm.bin'),
            #     dic=os.path.join(model_path, 'cmudict-en-us.dict')
            # )
            # for phrase in speech:
            #     print("phrase:", phrase)
            #     print(phrase.segments(detailed=True))
        print("-" * 30)


def get_duration_wav(file_path):
    """leojnjn
    contextlib获取wav音频文件时长
    :param file_path:
    :return:
    """
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def get_vowels(String):
    """leojnjn
    该片段将在字符串中找到元音“a e i o u”
    :param String:
    :return:
    """
    return [each for each in String if each in "aeiou"]


def get_vowels_cn(String):
    """leojnjn
    该片段将在中文拼音列表中找到元音“a e i o u”
    :param String:
    :return:
    """
    # string = ''
    # # each = []
    # for index in range(len(List)):
    #     string.join(List[index])
    list_pinyin = pypinyin.lazy_pinyin(String)
    str1 = ''.join(list_pinyin)  # 使用 str.join() 方法返回整串
    print(str1)
    return [each for each in str1 if each in "aeiou"]


@torch.no_grad()
def main():
    img = Image.open(f"data/images/{args.character}.png")
    img = img.convert('RGBA')
    IMG_WIDTH = 512
    wRatio = img.size[0] / IMG_WIDTH
    img = img.resize((IMG_WIDTH, int(img.size[1] / wRatio)))
    for i, px in enumerate(img.getdata()):
        if px[3] <= 0:
            y = i // IMG_WIDTH
            x = i % IMG_WIDTH
            img.putpixel((x, y), (0, 0, 0, 0))
    input_image = preprocessing_image(img.crop((0, 0, IMG_WIDTH, IMG_WIDTH)))
    if args.model.endswith('half'):
        input_image = torch.from_numpy(input_image).half() * 2.0 - 1
    else:
        input_image = torch.from_numpy(input_image).float() * 2.0 - 1
    input_image = input_image.unsqueeze(0)
    extra_image = None
    if img.size[1] > IMG_WIDTH:
        extra_image = np.array(img.crop((0, IMG_WIDTH, img.size[0], img.size[1])))

    print("Character Image Loaded:", args.character)
    cap = None

    output_fps = FPS()

    if not args.debug_input:

        if args.ifm is not None:
            client_process = IFMClientProcess()
            client_process.daemon = True
            client_process.start()
            print("IFM Service Running:", args.ifm)

        if args.mouse_input is not None:
            client_process = MouseClientProcess()
            client_process.daemon = True
            client_process.start()
            print("Mouse Input Running")

        """leojnjn20221229
        # 创建AIClientProcess对象
        """
        if args.AI_input is not None:
            client_process = AIClientProcess()
            client_process.daemon = True
            client_process.start()
            print("AI Input Running")

        else:

            if args.input == 'cam':
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                if ret is None:
                    raise Exception("Can't find Camera")
            else:
                cap = cv2.VideoCapture(args.input)
                frame_count = 0
                os.makedirs(os.path.join('dst', args.character, args.output_dir), exist_ok=True)
                print("Webcam Input Running")

    facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    if args.output_webcam:
        cam_scale = 1
        cam_width_scale = 1
        if args.anime4k:
            cam_scale = 2
        if args.alpha_split:
            cam_width_scale = 2
        cam = pyvirtualcam.Camera(width=args.output_w * cam_scale * cam_width_scale, height=args.output_h * cam_scale,
                                  fps=60,
                                  backend=args.output_webcam,
                                  fmt=
                                  {'unitycapture': pyvirtualcam.PixelFormat.RGBA, 'obs': pyvirtualcam.PixelFormat.RGB}[
                                      args.output_webcam])
        print(f'Using virtual camera: {cam.device}')

    a = None

    if args.anime4k:
        parameters = ac.Parameters()
        # enable HDN for ACNet
        parameters.HDN = True

        # a = ac.AC(
        #     managerList=ac.ManagerList([ac.CUDAManager(dID=0)]),
        #     type=ac.ProcessorType.Cuda_ACNet,
        # )

        a = ac.AC(
            managerList=ac.ManagerList([ac.OpenCLACNetManager(pID=0, dID=0)]),
            type=ac.ProcessorType.OpenCL_ACNet,
        )
        a.set_arguments(parameters)
        print("Anime4K Loaded")

    position_vector = [0, 0, 0, 1]

    pose_queue = []
    blender_data = create_default_blender_data()
    mouse_data = {
        'eye_l_h_temp': 0,
        'eye_r_h_temp': 0,
        'mouth_ratio': 0,
        'eye_y_ratio': 0,
        'eye_x_ratio': 0,
        'x_angle': 0,
        'y_angle': 0,
        'z_angle': 0,
    }

    model_output = None
    model_process = ModelClientProcess(input_image)
    model_process.daemon = True
    model_process.start()

    print("Ready. Close this console to exit.")

    """leojnjn20221201
    mouthStatusLeo 嘴部状态 0 关闭 1 打开
    mouthShapeLeo嘴部形状14a 15i 16u 17e 18o 19△
    eyeStatusLeo = 0 左眼开 eyeStatusReo = 0 右眼开
    timer 随机眨眼计时器"""
    mouthStatusLeo = 0
    mouthShapeLeo = 14
    eyeStatusLeo, eyeStatusReo = 0, 0  # 睁眼
    last_blink_start = time.perf_counter()  # 眨眼
    is_blinking = False  # 眨眼
    r_t = random.uniform(5, 10)  # 眨眼
    tic_rands = 0.0
    dlt_rands = 0.0

    """leojnjn20221208+
    # 创建声音到嘴型线程对象
    """
    threads = [Wav2MouthThread() for i in range(1)]  # , Aud2MouthThread()]  #
    for t in threads:
        lock = threading.Lock()  # 在线程同步执行获取数据的过程中，容易造成数据不同步现象，这时候可以给资源加锁
        lock.acquire()
        t.start()  # 启动
        lock.release()
    """leojnjn20221208+
    # 创建声音到嘴型子进程对象
    """
    # wav2mouth_process = Wav2MouthProcess()
    # wav2mouth_process.daemon = True
    # wav2mouth_process.start()

    while True:
        """leojnjn20221206
        # 线程实时调整嘴形 赋值mouse和webcam中的变量改变影响2者功能呈现
        mouthStatusLeo 嘴部状态 0 关闭 1 打开
        mouthShapeLeo嘴部形状14a 15i 16u 17e 18o 19△
        """
        mouthStatusLeo = np_volume  # ret_volume_queue[0]
        # mouthShapeLeo = ret_phrase_queue[0]
        if "a" == ret_phrase_queue[0]:
            # print("aaa")
            mouthShapeLeo = 14
        elif "e" == ret_phrase_queue[0]:
            # print("eee")
            mouthShapeLeo = 17
        elif "i" == ret_phrase_queue[0]:
            # print("ooo")
            mouthShapeLeo = 15
        elif "o" == ret_phrase_queue[0]:
            # print("uuu")
            mouthShapeLeo = 18
        elif "u" == ret_phrase_queue[0]:
            # print("iii")
            mouthShapeLeo = 16
        else:
            # print("!!!")
            mouthShapeLeo = 19
        # print(ret_phrase_queue[0])

        """★leojnjn 键盘监听
        需要注意的是必须使用cv加载图像，只有点击图像窗口才能侦听点击窗口时所使用的按键
        ord和chr的用法我这里重复一下，可以实现对于acall码的解释，方便直接看到按键结果
        ord()函数主要用来返回对应字符的ascii码，
        chr()主要用来表示ascii码对应的字符，可以用十进制，也可以用十六进制。
        说白了就是对键盘事件进行delay(50ms==0.05s)的等待（delay=0则为无限等待），若触发则返回该按键的ASSIC码（否则返回-1）
        """
        # k = cv2.waitKey(50)  # & 0xFF
        # # print(k)
        # if -1 == k:
        #     # print("^_^")
        #     eyeStatusReo, eyeStatusLeo = 0, 0
        # else:  # k != -1
        #     # print("^_^")
        #     if ord("t") == k:
        #         # print("-_-")
        #         eyeStatusReo, eyeStatusLeo = 1, 1
        #     else:
        #         if ord("r") == k:
        #             # print("-_^")
        #             eyeStatusReo = 1
        #         if ord("y") == k:
        #             # print("^_-")
        #             eyeStatusLeo = 1
        '''------------------------------------------------------------------------------------'''

        # ret, frame = cap.read()
        # input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = facemesh.process(input_frame)
        if args.perf == 'main':
            tic = time.perf_counter()
        if args.debug_input:
            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            pose_vector_c = [0.0] * 6

            mouth_eye_vector_c[2] = math.sin(time.perf_counter() * 3)
            mouth_eye_vector_c[3] = math.sin(time.perf_counter() * 3)

            mouth_eye_vector_c[14] = 0

            mouth_eye_vector_c[25] = math.sin(time.perf_counter() * 2.2) * 0.2
            mouth_eye_vector_c[26] = math.sin(time.perf_counter() * 3.5) * 0.8

            pose_vector_c[0] = math.sin(time.perf_counter() * 1.1)
            pose_vector_c[1] = math.sin(time.perf_counter() * 1.2)
            pose_vector_c[2] = math.sin(time.perf_counter() * 1.5)



        elif args.ifm is not None:
            # get pose from ifm
            try:
                new_blender_data = blender_data
                while not client_process.should_terminate.value and not client_process.queue.empty():
                    new_blender_data = client_process.queue.get_nowait()
                blender_data = new_blender_data
            except queue.Empty:
                pass

            ifacialmocap_pose_converted = ifm_converter.convert(blender_data)

            # ifacialmocap_pose = blender_data
            #
            # eye_l_h_temp = ifacialmocap_pose[EYE_BLINK_LEFT]
            # eye_r_h_temp = ifacialmocap_pose[EYE_BLINK_RIGHT]
            # mouth_ratio = (ifacialmocap_pose[JAW_OPEN] - 0.10)*1.3
            # x_angle = -ifacialmocap_pose[HEAD_BONE_X] * 1.5 + 1.57
            # y_angle = -ifacialmocap_pose[HEAD_BONE_Y]
            # z_angle = ifacialmocap_pose[HEAD_BONE_Z] - 1.57
            #
            # eye_x_ratio = (ifacialmocap_pose[EYE_LOOK_IN_LEFT] -
            #                ifacialmocap_pose[EYE_LOOK_OUT_LEFT] -
            #                ifacialmocap_pose[EYE_LOOK_IN_RIGHT] +
            #                ifacialmocap_pose[EYE_LOOK_OUT_RIGHT]) / 2.0 / 0.75
            #
            # eye_y_ratio = (ifacialmocap_pose[EYE_LOOK_UP_LEFT]
            #                + ifacialmocap_pose[EYE_LOOK_UP_RIGHT]
            #                - ifacialmocap_pose[EYE_LOOK_DOWN_RIGHT]
            #                + ifacialmocap_pose[EYE_LOOK_DOWN_LEFT]) / 2.0 / 0.75

            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            pose_vector_c = [0.0] * 6
            for i in range(0, 12):
                eyebrow_vector_c[i] = ifacialmocap_pose_converted[i]
            for i in range(12, 39):
                mouth_eye_vector_c[i - 12] = ifacialmocap_pose_converted[i]
            for i in range(39, 42):
                pose_vector_c[i - 39] = ifacialmocap_pose_converted[i]

            position_vector = blender_data[HEAD_BONE_QUAT]

        elif args.mouse_input is not None:
            try:
                new_blender_data = mouse_data
                while not client_process.queue.empty():
                    new_blender_data = client_process.queue.get_nowait()
                mouse_data = new_blender_data
            except queue.Empty:
                pass

            eye_l_h_temp = mouse_data['eye_l_h_temp']
            eye_r_h_temp = mouse_data['eye_r_h_temp']
            mouth_ratio = mouse_data['mouth_ratio']
            eye_y_ratio = mouse_data['eye_y_ratio']
            eye_x_ratio = mouse_data['eye_x_ratio']
            x_angle = mouse_data['x_angle']
            y_angle = mouse_data['y_angle']
            z_angle = mouse_data['z_angle']

            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            pose_vector_c = [0.0] * 6

            mouth_eye_vector_c[2] = eye_l_h_temp
            mouth_eye_vector_c[3] = eye_r_h_temp

            mouth_eye_vector_c[14] = mouth_ratio * 1.5  # 1.5  # 14 #★a 15i 16u 17e 18o 19△d mouthShapeLeo

            mouth_eye_vector_c[25] = eye_y_ratio
            mouth_eye_vector_c[26] = eye_x_ratio

            pose_vector_c[0] = x_angle
            pose_vector_c[1] = y_angle
            pose_vector_c[2] = z_angle

        elif args.AI_input is not None:
            try:
                new_blender_data = mouse_data
                while not client_process.queue.empty():
                    new_blender_data = client_process.queue.get_nowait()
                mouse_data = new_blender_data
            except queue.Empty:
                pass

            """★leojnjn 随机眨眼的计时器部分 s 赋值mouse中的变量改变只影响mouse功能呈现
            time.perf_counter()是已经持续的时间 (math.pi / 2.0 - 0.1)         # math.sin(time.perf_counter() * 2)"""

            # lower_case = int((math.pi / 2.0 - 0.05) * 1000)
            # # print(int((math.pi / 2.0 + 0.1) * 1000))
            # tic_rands = int(time.perf_counter() * 1000)
            # # print(tic_rands)
            # interval_milseconds = random.randint(0, 100)  # 1~60秒随机选一个时间
            # if (tic_rands % lower_case) < interval_milseconds:
            #     if math.sin(time.perf_counter()) > 0:
            #         eyeStatusReo, eyeStatusLeo = math.sin(time.perf_counter()), math.sin(time.perf_counter())
            # else:
            #     eyeStatusReo, eyeStatusLeo = 0, 0  # 睁眼
            # # print(eyeStatusReo)
            def blink_function(t_, ect):
                # 1-sqrt(2e)/(sqrt(3)-1)*(t/ect)*exp(-(t/ect)^2)*sin(t/ect)
                # ect = 1 / 6  # eye_closing_time
                constant = math.sqrt(2 * math.e) / (math.sqrt(3) - 1)
                return 1 - constant * (t_ / ect) * math.exp(-(t_ / ect) ** 2) * math.sin(t_ / ect)

            time_since_blink = (time.perf_counter() - last_blink_start)
            # print(time_since_blink)
            if not is_blinking:
                if time_since_blink > r_t:
                    last_blink_start = time.perf_counter()
                    time_since_blink = (time.perf_counter() - last_blink_start)
                    is_blinking = True
                    # print('start blinking')
                    r_t = random.uniform(5, 8)
                    # print(f'next blink in {r_t} seconds')
            if is_blinking:
                # print(time_since_blink)
                eyeStatusLeo = blink_function(time_since_blink, 0.2)
                eyeStatusReo = blink_function(time_since_blink, 0.2)
                # print(eye_l_h_temp)
            if is_blinking and time_since_blink > 0.2:
                is_blinking = False
                eyeStatusReo, eyeStatusLeo = 0, 0
            eye_l_h_temp = eyeStatusLeo  # mouse_data['eye_l_h_temp']# ★ 眼部状态 0 打开 1 关闭
            eye_r_h_temp = eyeStatusReo  # mouse_data['eye_r_h_temp']# ★ 眼部状态 0 打开 1 关闭

            mouth_ratio = mouthStatusLeo  # mouse_data['mouth_ratio']# ★ 嘴部状态 0 关闭 1 打开张嘴与否
            eye_y_ratio = mouse_data['eye_y_ratio']
            eye_x_ratio = mouse_data['eye_x_ratio']
            x_angle = mouse_data['x_angle']
            y_angle = mouse_data['y_angle']
            z_angle = mouse_data['z_angle']

            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            pose_vector_c = [0.0] * 6

            mouth_eye_vector_c[2] = eye_l_h_temp
            mouth_eye_vector_c[3] = eye_r_h_temp

            mouth_eye_vector_c[mouthShapeLeo] = mouth_ratio * 1.5  # 1.5  # 14 #★a 15i 16u 17e 18o 19△d mouthShapeLeo

            mouth_eye_vector_c[25] = eye_y_ratio
            mouth_eye_vector_c[26] = eye_x_ratio

            pose_vector_c[0] = x_angle
            pose_vector_c[1] = y_angle
            pose_vector_c[2] = z_angle

        else:  # 动捕
            ret, frame = cap.read()
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = facemesh.process(input_frame)

            if results.multi_face_landmarks is None:
                continue

            facial_landmarks = results.multi_face_landmarks[0].landmark

            if args.debug:
                pose, debug_image = get_pose(facial_landmarks, frame)
            else:
                pose = get_pose(facial_landmarks)

            if len(pose_queue) < 3:
                pose_queue.append(pose)
                pose_queue.append(pose)
                pose_queue.append(pose)
            else:
                pose_queue.pop(0)
                pose_queue.append(pose)

            np_pose = np.average(np.array(pose_queue), axis=0, weights=[0.6, 0.3, 0.1])

            eye_l_h_temp = np_pose[0]
            eye_r_h_temp = np_pose[1]
            mouth_ratio = mouthStatusLeo  # np_pose[2]
            eye_y_ratio = np_pose[3]
            eye_x_ratio = np_pose[4]
            x_angle = np_pose[5]
            y_angle = np_pose[6]
            z_angle = np_pose[7]

            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            pose_vector_c = [0.0] * 6

            mouth_eye_vector_c[2] = eye_l_h_temp
            mouth_eye_vector_c[3] = eye_r_h_temp

            mouth_eye_vector_c[mouthShapeLeo] = mouth_ratio * 1.5  # 1.5 # 14

            mouth_eye_vector_c[25] = eye_y_ratio
            mouth_eye_vector_c[26] = eye_x_ratio

            pose_vector_c[0] = (x_angle - 1.5) * 1.6
            pose_vector_c[1] = y_angle * 2.0  # temp weight
            pose_vector_c[2] = (z_angle + 1.5) * 2  # temp weight

        pose_vector_c[3] = pose_vector_c[1]
        pose_vector_c[4] = pose_vector_c[2]

        model_input_arr = eyebrow_vector_c
        model_input_arr.extend(mouth_eye_vector_c)
        model_input_arr.extend(pose_vector_c)

        model_process.input_queue.put_nowait(model_input_arr)

        has_model_output = 0
        try:
            new_model_output = model_output
            while not model_process.output_queue.empty():
                has_model_output += 1
                new_model_output = model_process.output_queue.get_nowait()
            model_output = new_model_output
        except queue.Empty:
            pass
        if model_output is None:
            time.sleep(1)
            continue
        # print(has_model_output)
        # should_output=should_output or has_model_output
        # if not should_output:
        #     continue

        postprocessed_image = model_output

        if args.perf == 'main':
            print('===')
            print("input", time.perf_counter() - tic)
            tic = time.perf_counter()

        if extra_image is not None:
            postprocessed_image = cv2.vconcat([postprocessed_image, extra_image])

        k_scale = 1
        rotate_angle = 0
        dx = 0
        dy = 0
        if args.extend_movement:
            k_scale = position_vector[2] * math.sqrt(args.extend_movement) + 1
            rotate_angle = -position_vector[0] * 10 * args.extend_movement
            dx = position_vector[0] * 400 * k_scale * args.extend_movement
            dy = -position_vector[1] * 600 * k_scale * args.extend_movement
        if args.bongo:
            rotate_angle -= 5
        rm = cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_WIDTH / 2), rotate_angle, k_scale)
        rm[0, 2] += dx + args.output_w / 2 - IMG_WIDTH / 2
        rm[1, 2] += dy + args.output_h / 2 - IMG_WIDTH / 2

        postprocessed_image = cv2.warpAffine(
            postprocessed_image,
            rm,
            (args.output_w, args.output_h))

        if args.perf == 'main':
            print("extendmovement", (time.perf_counter() - tic) * 1000)
            tic = time.perf_counter()

        output_fps_number = output_fps()

        if args.anime4k:
            alpha_channel = postprocessed_image[:, :, 3]
            alpha_channel = cv2.resize(alpha_channel, None, fx=2, fy=2)

            # a.load_image_from_numpy(cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2RGB), input_type=ac.AC_INPUT_RGB)
            # img = cv2.imread("character/test41.png")
            img1 = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGR)
            # a.load_image_from_numpy(img, input_type=ac.AC_INPUT_BGR)
            a.load_image_from_numpy(img1, input_type=ac.AC_INPUT_BGR)
            a.process()
            postprocessed_image = a.save_image_to_numpy()
            postprocessed_image = cv2.merge((postprocessed_image, alpha_channel))
            postprocessed_image = cv2.cvtColor(postprocessed_image, cv2.COLOR_BGRA2RGBA)
            if args.perf == 'main':
                print("anime4k", (time.perf_counter() - tic) * 1000)
                tic = time.perf_counter()
        if args.alpha_split:
            alpha_image = cv2.merge(
                [postprocessed_image[:, :, 3], postprocessed_image[:, :, 3], postprocessed_image[:, :, 3]])
            alpha_image = cv2.cvtColor(alpha_image, cv2.COLOR_RGB2RGBA)
            postprocessed_image = cv2.hconcat([postprocessed_image, alpha_image])

        if args.debug:
            output_frame = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGRA)
            # resized_frame = cv2.resize(output_frame, (np.min(debug_image.shape[:2]), np.min(debug_image.shape[:2])))
            # output_frame = np.concatenate([debug_image, resized_frame], axis=1)
            cv2.putText(output_frame, str('OUT_FPS:%.1f' % output_fps_number), (0, 16), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f / %.1f' % (model_process.model_fps_number.value, model_process.gpu_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            else:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f' % (model_process.model_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.ifm is not None:
                cv2.putText(output_frame, str('IFM_FPS:%.1f' % client_process.ifm_fps_number.value), (0, 48),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str('MEMCACHED:%.1f%%' % (model_process.cache_hit_ratio.value * 100)),
                            (0, 64),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_gpu_cache_len > 0:
                cv2.putText(output_frame, str('GPUCACHED:%.1f%%' % (model_process.gpu_cache_hit_ratio.value * 100)),
                            (0, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow("frame", output_frame)
            # cv2.imshow("camera", debug_image)
            cv2.waitKey(1)
        if args.output_webcam:
            # result_image = np.zeros([720, 1280, 3], dtype=np.uint8)
            # result_image[720 - 512:, 1280 // 2 - 256:1280 // 2 + 256] = cv2.resize(
            #     cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2RGB), (512, 512))
            result_image = postprocessed_image
            if args.output_webcam == 'obs':
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
            cam.send(result_image)
            cam.sleep_until_next_frame()
        if args.perf == 'main':
            print("output", (time.perf_counter() - tic) * 1000)
            tic = time.perf_counter()

        # """★leojnjn 键盘监听
        # 需要注意的是必须使用cv加载图像，只有点击图像窗口才能侦听点击窗口时所使用的按键
        # ord和chr的用法我这里重复一下，可以实现对于acall码的解释，方便直接看到按键结果
        # ord()函数主要用来返回对应字符的ascii码，
        # chr()主要用来表示ascii码对应的字符，可以用十进制，也可以用十六进制。
        # 说白了就是对键盘事件进行delay(50ms==0.1s)的等待（delay=0则为无限等待），若触发则返回该按键的ASSIC码（否则返回-1）
        # """
        # k = cv2.waitKey(100)  # & 0xFF
        # # print(k)
        # if k == -1:
        #     # print("^_^")
        #     mouthStatusLeo = 0
        # else:  # k != -1
        #     # print("^o^")
        #     mouthStatusLeo = .7  # 根据不同角色嘴大小微调
        #     if k == ord("a"):
        #         # print("aaa")
        #         mouthShapeLeo = 14
        #     elif k == ord("e"):
        #         # print("eee")
        #         mouthShapeLeo = 17
        #     elif k == ord("o"):
        #         # print("ooo")
        #         mouthShapeLeo = 18
        #     elif k == ord("u"):
        #         # print("uuu")
        #         mouthShapeLeo = 16
        #     elif k == ord("i"):
        #         # print("iii")
        #         mouthShapeLeo = 15
        #     else:
        #         # print("!!!")
        #         mouthShapeLeo = 19
        # """leojnjn20221208+
        # # 创建声音到嘴型线程对象
        # 为了保证子线程在主线程执行结束前优先结束，可以采取join方法，来保证子线程执行完后才会结束主线程。"""
        # for t in threads:
        #     t.join()


if __name__ == '__main__':
    main()
