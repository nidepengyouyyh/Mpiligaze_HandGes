from openai import OpenAI
import base64
import numpy as np
import cv2
import pyaudio
import time

from agent.config import Config

class MultimodalAssistant:
    def __init__(self, api_key, base_url, model="qwen-omni-turbo", voice="Cherry"):
        """
        初始化多模态助手
        :param api_key: API密钥
        :param base_url: 服务端点
        :param model: 模型名称 (默认qwen-omni-turbo)
        :param voice: 语音类型 (默认Cherry)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.audio_config = {
            "voice": voice,
            "format": "wav"
        }
        self.p = None
        self.stream = None

        self.history = [{
                    "role": "system",
                    "content": [{"type": "text",
                                 "text": "你现在是一个ai语音助手，当提及图片有关内容的时候你才需要提及图片内容，其他时候必须正常回答文字问题"}],
                }]

        self.summary_prompt = [{
            "role": "system",
            "content": Config.LLM_SUMMARY_PROMPT
        },
            {"role": "user",
             "content": None}]

    @staticmethod
    def capture_image(output_path="../agent/img/capture.jpg"):
        """
        捕获摄像头图像
        :param output_path: 图像保存路径
        :return: 图像路径或None
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(">>>摄像头访问失败")

        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(">>>图像捕获失败")

            cv2.imwrite(output_path, frame)
            return output_path
        finally:
            cap.release()

    @staticmethod
    def _encode_image(image_path):
        """
        图像Base64编码
        :param image_path: 图像路径
        :return: Base64编码字符串
        """
        with open(image_path, 'rb') as f:
            return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode('utf-8')

    def generate(self, text:str, img, check:bool=True):
        self.set_summary()
        his = []
        self.history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img},
                        },
                        {"type": "text", "text": text},
                    ],
                })

        print(self.history)
        # noinspection PyTypeChecker
        completion = self.client.chat.completions.create(
            model="qwen-omni-turbo",
            messages=self.history,
            # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
            modalities=["text", "audio"],
            audio=self.audio_config,
            # stream 必须设置为 True，否则会报错
            stream=True,
            stream_options={
                "include_usage": True
            }
        )
        self.p = pyaudio.PyAudio()
        # 创建音频流
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True)
        if check:
            print(">> 语音助手: ", end="")
        for chunk in completion:
            if chunk.choices:
                if hasattr(chunk.choices[0].delta, "audio"):
                    try:
                        audio_string = chunk.choices[0].delta.audio["data"]
                        wav_bytes = base64.b64decode(audio_string)
                        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                        # 直接播放音频数据
                        self.stream.write(audio_np.tobytes())
                    except Exception as e:
                        print(chunk.choices[0].delta.audio["transcript"], end="")
                        his.append(chunk.choices[0].delta.audio["transcript"])
        time.sleep(0.8)
        result = "".join(his)
        self.history.append({
                    "role": "assistant",
                    "content": [{"type": "text",
                                 "text": result}],
                })
        print()

    def clean(self):
        # 清理资源
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    @staticmethod
    def get_time():
        current_time = time.localtime()
        formatted_time = time.strftime("%Y年%m月%d日%H时%M分%S秒", current_time)
        return formatted_time

    def summary(self, text):
        time_now = self.get_time()
        time_text = time_now + "：" + text
        self.summary_prompt[1]["content"] = time_text
        response = self.client.chat.completions.create(
            model="qwen-max-latest",
            messages=self.summary_prompt,
            stream=False
        )
        summary = response.choices[0].message.content
        if summary != "False":
            self.save_summary(summary)

    def set_summary(self):
        summary = self.get_summary()
        self.history[0]["content"][0]["text"] = self.history[0]["content"][0]["text"] + summary

    @staticmethod
    def get_summary():
        summary = []
        with open("../agent/temp/summary.txt", "r", encoding="ISO-8859-1") as f:
            for line in f:
                summary.append(line.strip())
        return f"{summary}"

    @staticmethod
    def save_summary(summary):
        with open("../agent/temp/summary.txt", "a") as f:
            f.write(f"{summary}\n")

if __name__=="__main__":
    agent = MultimodalAssistant(api_key=Config.QWEN_API_KEY, base_url=Config.QWEN_URL)
    agent.generate("画面前的帅哥你喜欢吗,他的生日是1月2日")
    agent.generate("刚刚说他的生日是什么时候来着")