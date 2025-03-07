class Config:
    # 语音识别配置
    VOSK_MODEL_PATH = r"vosk/vosk-model-cn-0.22"

    # 大语言模型配置(OpenAI兼容)
    LLM_API_KEY = "sk-73eef6dafbc747168fbfe6a9376eefd8"
    LLM_BASE_URL = "https://api.deepseek.com"
    LLM_MODEL = "deepseek-chat"

    LLM_API_KEY_1 = "sk-soilybcvoykmnucazobiwaaqsrdtskfjjqnxychuinevyacm"
    LLM_BASE_URL_1 = "https://api.siliconflow.cn/v1"
    LLM_MODEL_1 = "deepseek-ai/DeepSeek-V3"

    LLM_API_KEY_2 = "sk-03cb872fee534bfe8a4284329b73da5b"
    LLM_BASE_URL_2 = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_2 = "qwen-max-latest"

    LLM_SYSTEM_PROMPT = "你是桌面助手小白，用简洁口语化的中文回答"
    LLM_SUMMARY_PROMPT = '''
    你是一位心思细腻善于记录人生活需求的总结专家，你必须严格按照以下规则输出：
    1.分析用户的话，提取其中的重要事件
    2.将事件提取出来，并且解析为一个以年月日时分秒为key的字典
    3.如果没有类似的重点信息就回复"False"
    
    EXAMPLE:
    Command: "2024年12月04日12时53分24秒：我明天要吃药了"
    Output: {"2024年12月04日12时53分24秒":"用户明天要吃药"}
    
    Command: "2025年01月04日10时51分45秒：今天我女神李静怡给我表白了，好开心"
    Output: {"2025年01月04日10时51分45秒":"用户的女神是李静怡，用户今天被女神表白了"}
    
    Command: "2023年11月05日09时01分45秒：我的生日是十月六日"
    Output: {"2025年11月05日09时01分45秒":"用户的生日是十月六日"}
    
    Command: "2022年10月05日08时01分45秒：屏幕前有什么"
    Output: False
    
    Command: "2025年02月25日23时43分15秒：我生日是什么时候"
    Output: False
    
    Command:"2025年02月25日23时43分32秒：给我唱生日歌"
    Output: False
    
    只回复字典本身即可，不要回复其它内容
    
    Current Command:
    '''

    # 语音合成配置(用的是阿里云的cosyvoice)
    TTS_API_KEY = "sk-03cb872fee534bfe8a4284329b73da5b"
    TTS_MODEL = "cosyvoice-v1"
    TTS_VOICE = "longxiaochun"
    TTS_FORMAT = "PCM_22050HZ_MONO_16BIT"

    # 视觉模型配置(用的是阿里云的qwen_vl, 兼容OpenAI框架)
    VLM_API_KEY = "sk-03cb872fee534bfe8a4284329b73da5b"
    VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    VLM_MODEL = "qwen-vl-max-latest"
    VLM_SYSTEM_PROMPT = '''
    我即将说一句给机械臂的指令，你帮我从这句话中提取出起始物体和终止物体，并从这张图中分别找到这两个物体左上角和右下角的像素坐标，输出json数据结构。

    例如，如果我的指令是：请帮我把红色方块放在房子简笔画上。
    你输出这样的格式：
    {
     "start":"红色方块",
     "start_xyxy":[[102,505],[324,860]],
     "end":"房子简笔画",
     "end_xyxy":[[300,150],[476,310]]
    }

    只回复json本身即可，不要回复其它内容

    我现在的指令是：
    '''# 系统提示词——参考自同济子豪兄(https://github.com/TommyZihao/vlm_arm)

    #ollama配置
    OLLAMA_MODEL = "qwen2.5"
    OLLAMA_URL = "http://localhost:11434/api/chat"

    # qwen多模态
    QWEN_API_KEY = "sk-03cb872fee534bfe8a4284329b73da5b"
    QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL = "qwen-omni-turbo"
    QWEN_VOICE = "Cherry"