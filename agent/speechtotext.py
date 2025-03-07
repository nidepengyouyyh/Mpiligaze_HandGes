import json
import queue
import sys

import pyaudio
from vosk import Model, KaldiRecognizer
from typing import Union

from agent.texttotext import MultiModalDialogueManager
from agent.ai_helper import MultimodalAssistant

import threading

class SpeechRecognizer:
    """稳定版语音识别模块（非流式）"""

    def __init__(self, model_path: str, dialogue_manager:Union[MultiModalDialogueManager, MultimodalAssistant], model_type:str="normal"):
        """
        初始化语音识别器

        参数:
            model_path (str): Vosk模型路径
            dialogue_manager: 对话管理系统实例  # 修正参数说明
            model_type: normal 或者 vlm
        """
        # 初始化语音识别模型
        self.model = Model('../agent/' + model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.model_type = model_type

        # 关联对话系统（统一属性名称）
        self.dialogue_manager = dialogue_manager

        # 音频流配置
        self._chunk_size = 4000
        self._format = pyaudio.paInt16
        self._channels = 1
        self._rate = 16000

        # 初始化音频设备
        self._p = None
        self._stream = None
        self._init_audio_stream()

        #输入提示判断
        self.show_input_prompt = True

        self.img_queue = queue.Queue()
        #识别模式切换
        self.current_mode = 'nomal'

    def set_mode(self, mode):
            self.current_mode = mode
            # print(self.current_mode)
    def get_mode(self):
            return self.current_mode

    def start_background_task(self):
        """在新线程中启动语音识别"""
        threading.Thread(target=self.run, daemon=True).start()
    def add_img_to_queue(self, img):
        self.img_queue.put(img)

    def _init_audio_stream(self):
        """初始化音频输入流"""
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            start=False
        )

    def _cleanup_audio_resources(self):
        """清理音频资源"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._p.terminate()

    def _process_audio_chunk(self, data: bytes, img):
        """
        处理音频数据块

        返回:
            bool: 是否收到退出指令
        """
        self.show_input_prompt = False

        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            text = result.get('text', '').replace(" ", "")

            if self._is_exit_command(text):
                print(text)
                print(">> 收到退出指令")
                self.set_mode('exit')
                return True

            if self._is_hand(text):
                print(text)
                print(">> 切换手势识别")
                self.set_mode('hand')
                return True

            if self._is_eyes(text):
                print(text)
                print(">> 切换眼神识别")
                self.set_mode('eye')
                return True

            if text:
                print(f"{text}")
                self.dialogue_manager.summary(text)
                if self.model_type == "normal":
                    self._handle_recognized_text(text)
                elif self.model_type == "vlm":
                    self._handle_recognized_text_vlm(text, img)
                self.show_input_prompt = True

        return False

    def _handle_recognized_text(self, text: str):
        """统一方法调用名称"""
        try:
            self.dialogue_manager.process_stream_response(text)
        except Exception as e:
            print(f"对话处理失败: {str(e)}")

    def _handle_recognized_text_vlm(self, text: str, img:str):
        """统一方法调用名称"""
        try:
            self.dialogue_manager.generate(text, img)
        except Exception as e:
            print(f"对话处理失败: {str(e)}")

    @staticmethod
    def _is_exit_command(text: str) -> bool:
        """检测退出指令"""
        return any(cmd in text for cmd in ['退出', '结束', '停止', '关闭'])

    def run(self):
        """启动语音识别主循环"""
        print(">> 语音助手启动中...")
        self._stream.start_stream()
        while True:
                if self.show_input_prompt:
                    print(">> 用户输入:", end="")

                # 从队列中获取最新的一帧图像
                img = None
                if not self.img_queue.empty():
                    img = self.img_queue.get()

                data = self._stream.read(self._chunk_size)
                form = self._process_audio_chunk(data, img)


    @staticmethod
    def _is_hand(text: str) -> bool:
        """检测退出指令"""
        return any(cmd in text for cmd in ['手势识别','切换手势'])

    @staticmethod
    def _is_eyes(text: str) -> bool:
        """检测退出指令"""
        return any(cmd in text for cmd in ['眼神识别', '切换眼神', '无障碍','视线'])
