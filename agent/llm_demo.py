from threading import Lock
from typing import Optional

from agent.config import Config
from agent.textspeech import TTSStreamManager
from agent.texttotext import MultiModalDialogueManager
from agent.speechtotext import SpeechRecognizer
# from new_yolo import VisionProcessor
from agent.ai_helper import MultimodalAssistant

# ========== 显式声明全局变量 ==========
_vision: Optional['VisionProcessor'] = None
_tts: Optional['TTSStreamManager'] = None
_dialogue: Optional['MultiModalDialogueManager'] = None
_speech: Optional['SpeechRecognizer'] = None
_dialogue_vlm : Optional['MultimodalAssistant'] = None
check = False

# ========== 初始化锁 ==========
_vision_lock = Lock()
_tts_lock = Lock()
_dialogue_lock = Lock()
_speech_lock = Lock()
_dialogue_vlm_lock = Lock()


# ========== 初始化方法 ==========
def _init_vision() -> 'VisionProcessor':
    """初始化视觉模块"""
    global _vision
    with _vision_lock:
        if _vision is None:
            from agent.new_yolo import VisionProcessor  # 延迟导入
            _vision = VisionProcessor("yl/yolo11s.pt")
            _vision.start_detection_loop()
    return _vision


def _init_tts() -> 'TTSStreamManager':
    """初始化语音合成"""
    global _tts
    with _tts_lock:
        if _tts is None:
            from agent.textspeech import TTSStreamManager  # 延迟导入
            _tts = TTSStreamManager(
                api_key=Config.TTS_API_KEY,
                model=Config.TTS_MODEL,
                voice=Config.TTS_VOICE,
                format_name=Config.TTS_FORMAT
            )
    return _tts


def _init_dialogue() -> 'MultiModalDialogueManager':
    """初始化对话系统"""
    global _dialogue
    with _dialogue_lock:
        if _dialogue is None:
            from agent.texttotext import MultiModalDialogueManager  # 延迟导入
            # 初始化依赖
            vision = _init_vision()
            tts = _init_tts()

            _dialogue = MultiModalDialogueManager(
                vision_processor=vision,
                api_key=Config.LLM_API_KEY_2,
                base_url=Config.LLM_BASE_URL_2,
                model=Config.LLM_MODEL_2,
                tts_service=tts
            )
    return _dialogue

def _init_dialogue_vlm() -> 'MultimodalAssistant':
    """初始化对话系统"""
    global _dialogue_vlm
    with _dialogue_vlm_lock:
        if _dialogue_vlm is None:
            from agent.ai_helper import MultimodalAssistant  # 延迟导入
            # 初始化依赖
            _dialogue_vlm = MultimodalAssistant(
                api_key=Config.QWEN_API_KEY,
                base_url=Config.QWEN_URL,
                model=Config.QWEN_MODEL
            )
    return _dialogue_vlm


def _init_speech() -> 'SpeechRecognizer':
    """初始化语音识别"""
    global _speech
    with _speech_lock:
        if _speech is None:
            from agent.speechtotext import SpeechRecognizer  # 延迟导入
            dialogue = _init_dialogue()

            _speech = SpeechRecognizer(
                model_path=Config.VOSK_MODEL_PATH,
                dialogue_manager=dialogue
            )
    return _speech


def _init_speech_vlm() -> 'SpeechRecognizer':
    """初始化语音识别"""
    global _speech
    with _speech_lock:
        if _speech is None:
            from agent.speechtotext import SpeechRecognizer  # 延迟导入
            dialogue = _init_dialogue_vlm()

            _speech = SpeechRecognizer(
                model_path=Config.VOSK_MODEL_PATH,
                dialogue_manager=dialogue,
                model_type="vlm"
            )
            _speech.start_background_task()
    return _speech


# ========== 用户接口 ==========
def speak(text: str):
    """语音合成（自动初始化TTS）"""
    tts = _init_tts()  # 正确获取实例
    tts.start_stream("default")
    tts.stream_text(text, "default")
    tts.complete_stream("default")


def chat(text: str, use_voice: bool = False) -> str:
    """对话接口"""
    dialogue = _init_dialogue()

    if use_voice:
        return dialogue.process_stream_response(text)
    else:
        return dialogue.text_only_response(text)


def chat_vlm(text:str):
    dialog_vlm = _init_dialogue_vlm()
    dialog_vlm.generate(text)
    if check:
        dialog_vlm.clean()


def listen():
    """语音监听"""
    global check
    check = True
    speech = _init_speech()
    speech.run()

def listen_vlm():
    """语音监听"""
    global check
    check = True
    speech = _init_speech_vlm()
    return speech


if __name__ == "__main__":
    listen_vlm()  # 应加载语音识别模块
    speak('你好')
