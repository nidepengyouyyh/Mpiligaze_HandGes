import re
import time
from openai import OpenAI

from agent.config import Config
from agent.new_yolo import VisionProcessor

class StreamDialogueManager:
    """
    流式对话管理核心类

    关键修复点：
    1. 统一方法名称为 process_stream_response
    2. 完善TTS会话生命周期管理
    3. 增强异常处理
    """

    def __init__(self, api_key: str, base_url: str, model: str, tts_service):
        # 初始化OpenAI客户端
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.tts = tts_service

        # 对话历史管理
        self.history = [{
            "role": "system",
            "content": Config.LLM_SYSTEM_PROMPT
        }]

        #总结提示词
        self.summary_prompt = [{
            "role": "system",
            "content": Config.LLM_SUMMARY_PROMPT
        },
            {"role":"user",
             "content":None}]

        # 流式控制参数
        self.current_session_id = None
        self.MIN_CHUNK_LENGTH = 8  # 最小发送字符数
        self.MAX_CHUNK_LENGTH = 30  # 最大缓冲字符数

    @staticmethod
    def get_time():
        current_time = time.localtime()
        formatted_time = time.strftime("%Y年%m月%d日%H时%M分%S秒", current_time)
        return formatted_time

    def summary(self, text):
        time_now = self.get_time()
        time_text = time_now + "：" + text
        self.summary_prompt[1]["content"] = time_text
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=self.summary_prompt,
            stream=False
        )
        summary = response.choices[0].message.content
        if summary != "False":
            self.save_summary(summary)

    def set_summary(self):
        summary = self.get_summary()
        self.history[0]["content"] = self.history[0]["content"] + summary

    @staticmethod
    def get_summary():
        summary = []

        with open("../agent/temp/summary.txt", "r") as f:
            for line in f:
                summary.append(line.strip())
        return f"{summary}"

    @staticmethod
    def save_summary(summary):
        with open("../agent/temp/summary.txt", "a") as f:
            f.write(f"{summary}\n")

    def process_stream_response(self, user_input: str):
        """处理流式响应的入口方法（修正方法名）"""
        try:
            # 启动新会话
            self._start_new_session()

            self.set_summary()

            # 记录用户输入
            self._add_to_history(user_input, "user")

            # 获取LLM流式响应
            llm_stream = self._get_llm_stream()

            # 处理流式内容
            full_response = self._process_llm_stream(llm_stream)

            # 记录完整响应
            self._add_to_history(full_response, "assistant")

        except Exception as e:
            print(f"对话处理失败: {str(e)}")
            raise
        finally:
            self._cleanup_session()

    def _start_new_session(self):
        """初始化新TTS会话"""
        self.current_session_id = f"session_{int(time.time() * 1000)}"
        self.tts.start_stream(self.current_session_id)

    def _get_llm_stream(self):
        """获取LLM流式响应"""
        try:
            return self.llm_client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=True,
                temperature=0.7,
                max_tokens=512,
                top_p=0.7
            )
        except Exception as e:
            raise RuntimeError(f"LLM请求失败: {str(e)}")

    def _process_llm_stream(self, stream) -> str:
        """处理流式响应内容（增加实时输出）"""
        buffer = []
        full_response = []

        print(">> AI回复: ", end="", flush=True)  # 初始化输出

        for chunk in stream:
            content = self._extract_content(chunk)
            if not content:
                continue

            buffer.append(content)
            full_response.append(content)

            # 实时输出每个字符
            print(content, end="", flush=True)

            # 智能分块逻辑
            if self._should_send_chunk(buffer):
                self._send_to_tts(buffer)
                buffer.clear()

        # 处理剩余内容
        if buffer:
            self._send_to_tts(buffer)

        print("")  # 结束输出
        return ''.join(full_response)

    def _extract_content(self, chunk) -> str:
        """从响应块提取并净化文本"""
        if not chunk.choices:
            return ""

        content = chunk.choices[0].delta.content or ""
        return self._sanitize_text(content)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """文本净化处理"""
        # 移除控制字符和异常空格
        text = re.sub(r'[\x00-\x1F\u200b-\u200f\ufeff]', '', text)
        # 替换非法字符
        return text.replace('�', '？').strip()

    def _should_send_chunk(self, buffer: list) -> bool:
        """判断是否发送当前缓冲"""
        current_text = ''.join(buffer)
        sentence_enders = {'。', '！', '？', '；', '\n', '，'}

        return (
                len(current_text) >= self.MAX_CHUNK_LENGTH or
                (len(current_text) >= self.MIN_CHUNK_LENGTH and
                 current_text[-1] in sentence_enders)
        )

    def _send_to_tts(self, buffer: list):
        """发送文本块到TTS"""
        text_chunk = ''.join(buffer)
        try:
            self.tts.stream_text(text_chunk, self.current_session_id)
        except Exception as e:
            print(f"TTS传输失败: {str(e)}")
            self._recover_tts_session()

    def _recover_tts_session(self):
        """恢复TTS会话"""
        print("尝试恢复TTS流...")
        self._cleanup_session()
        self._start_new_session()

    def _add_to_history(self, text: str, role: str):
        """记录对话历史"""
        if not text:
            return

        self.history.append({
            "role": role,
            "content": f"{text}"
        })

        # 保持历史记录长度
        if len(self.history) > 20:
            self.history = [self.history[0]] + self.history[-19:]

    def _cleanup_session(self):
        """清理会话资源"""
        if self.current_session_id:
            self.tts.complete_stream(self.current_session_id)
            self.current_session_id = None

    def text_only_response(self, user_input: str) -> str:
        """纯文本对话（不触发语音合成）"""
        try:
            # 记录对话历史
            self._add_to_history(user_input, "user")

            # 获取非流式响应
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=False  # 禁用流式
            )
            full_text = response.choices[0].message.content

            # 记录回复
            self._add_to_history(full_text, "assistant")
            return full_text

        except Exception as e:
            print(f"文本生成失败: {str(e)}")
            raise

class MultiModalDialogueManager(StreamDialogueManager):
    def __init__(self, vision_processor: VisionProcessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision = vision_processor

    def _get_llm_stream(self):
        """重写LLM请求方法以注入视觉信息"""
        dynamic_prompt = {
            "role": "system",
            "content": f"{Config.LLM_SYSTEM_PROMPT}\n\n{self._get_live_context()}"
        }

        # 临时插入动态上下文
        messages = [self.history[0], dynamic_prompt] + self.history[1:]

        return self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=512
        )

    def _get_live_context(self) -> str:
        """生成多模态上下文"""
        return (
            f"[实时环境感知]\n"
            f"1. 视觉信息：{self.vision.get_visual_context()}\n"
            f"2. 当前时间：{time.strftime('%Y-%m-%d %H:%M')}\n"
            f"3. 对话历史：最近{len(self.history) // 2}轮交流"
        )