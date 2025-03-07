import pyaudio
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback


class TTSStreamManager:
    """遵循官方规范的流式TTS管理器"""

    def __init__(self, api_key: str, model: str, voice: str, format_name: str):
        dashscope.api_key = api_key
        self.model = model
        self.voice = voice
        self.format = getattr(AudioFormat, format_name)
        self.sample_rate = self._parse_sample_rate(format_name)

        # 音频设备初始化
        self.player = pyaudio.PyAudio()
        self.active_streams = {}  # session_id: {'synthesizer': , 'stream': }

    def _parse_sample_rate(self, format_name: str) -> int:
        """从格式名称解析采样率"""
        return int(format_name.split('_')[1].replace('HZ', ''))

    def _get_audio_stream(self, session_id: str):
        """获取或创建音频输出流"""
        if session_id not in self.active_streams:
            stream = self.player.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True
            )
            self.active_streams[session_id] = {
                'stream': stream,
                'synthesizer': None,
                'callback': None
            }
        return self.active_streams[session_id]

    def start_stream(self, session_id: str):
        """初始化流式会话"""
        session = self._get_audio_stream(session_id)

        # 创建新的回调实例
        session['callback'] = StreamCallback(session['stream'])

        # 初始化合成器
        session['synthesizer'] = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=self.format,
            callback=session['callback']
        )

    def stream_text(self, text: str, session_id: str = "default"):
        """发送流式文本块"""
        session = self._get_audio_stream(session_id)
        if not session['synthesizer']:
            self.start_stream(session_id)

        try:
            session['synthesizer'].streaming_call(text)
        except Exception as e:
            print(f"流式传输失败: {str(e)}")
            self.close_stream(session_id)
            raise

    def complete_stream(self, session_id: str = "default"):
        """完成当前流式会话"""
        if session_id in self.active_streams:
            try:
                self.active_streams[session_id]['synthesizer'].streaming_complete()
            except Exception as e:
                print(f"完成流时出错: {str(e)}")
            finally:
                self.close_stream(session_id)

    def close_stream(self, session_id: str = "default"):
        """关闭指定会话"""
        if session_id in self.active_streams:
            session = self.active_streams.pop(session_id)
            session['callback'].close()
            session['stream'].stop_stream()
            session['stream'].close()


class StreamCallback(ResultCallback):
    """官方推荐的回调实现"""

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def on_open(self):
        pass
        #print(">> 音频流通道已建立")

    def on_data(self, data: bytes):
        try:
            self.stream.write(data)
        except OSError as e:
            print(f"!! 音频写入失败: {str(e)}")

    def on_complete(self):
        pass
        #print(">> 语音合成任务完成")

    def on_error(self, message: str):
        print(f"!! 合成错误: {message}")

    def on_close(self):
        pass
        #print(">> 音频流已关闭")

    def close(self):
        """主动关闭资源"""
        self.stream.stop_stream()
        self.stream.close()
