"""
导入视觉大模型模块
暂定使用阿里云
"""
import json
import base64
import cv2
from openai import OpenAI
from config import Config

SYSTEM_PROMPT = Config.VLM_SYSTEM_PROMPT

def qwen_vl_max_last(prompt, img_path):
    """
    :param prompt: 用户输入的内容
    :param img_path: 拍摄的图片路径
    :return: check:回复格式是否正确 content:整理后的结果..
    """
    client = OpenAI(
        api_key=Config.VLM_API_KEY,
        base_url=Config.VLM_BASE_URL,
    )

    # 编码为base64数据
    with open(img_path, 'rb') as image_file:
        image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')

    #请求数据
    completion = client.chat.completions.create(
        model=Config.VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT + prompt},
            {"type": "image_url","image_url": {"url": image}}
        ]}]
    )
    # 获取并解析JSON字符串
    json_str = completion.model_dump_json()

    # 使用json.loads解析JSON字符串
    data = json.loads(json_str)

    # 提取并清理实际内容
    message_content = data['choices'][0]['message']['content']
    try:
        # 移除代码块标记（```json 和 ```）
        if message_content.startswith('```json\n'):
            message_content = message_content[len('```json\n'):]
        if message_content.endswith('\n```'):
            message_content = message_content[:-len('\n```')]

        # 去除多余的换行符和缩进
        message_content = message_content.strip()

        # 解析清理后的JSON字符串
        parsed_message_content = json.loads(message_content)
        return True, parsed_message_content

    except:
        return False, None

def get_center(contents, facter, img_path):
    """
    :param contents: qwen_vl_max_last方法得到的结果
    :param facter: 缩放因子
    :param img_path: 拍摄图片的路径
    :return: 起始物品的中心点坐标x和y  结束物品的中心点坐标x和y
    """
    # 后处理
    img_bgr = cv2.imread(img_path)
    img_h = img_bgr.shape[0]
    img_w = img_bgr.shape[1]
    # 起点，左上角像素坐标
    start_x_min = int(contents['start_xyxy'][0][0] * img_w / facter)
    start_y_min = int(contents['start_xyxy'][0][1] * img_h / facter)
    # 起点，右下角像素坐标
    start_x_max = int(contents['start_xyxy'][1][0] * img_w / facter)
    start_y_max = int(contents['start_xyxy'][1][1] * img_h / facter)
    # 起点，中心点像素坐标
    start_x_center = int((start_x_min + start_x_max) / 2)
    start_y_center = int((start_y_min + start_y_max) / 2)
    # 终点，左上角像素坐标
    end_x_min = int(contents['end_xyxy'][0][0] * img_w / facter)
    end_y_min = int(contents['end_xyxy'][0][1] * img_h / facter)
    # 终点，右下角像素坐标
    end_x_max = int(contents['end_xyxy'][1][0] * img_w / facter)
    end_y_max = int(contents['end_xyxy'][1][1] * img_h / facter)
    # 终点，中心点像素坐标
    end_x_center = int((end_x_min + end_x_max) / 2)
    end_y_center = int((end_y_min + end_y_max) / 2)

    return start_x_center, start_y_center, end_x_center, end_y_center

if __name__ == '__main__':
    check, content = qwen_vl_max_last('帮我把红色方块放在二维码上', r'img/vl_now.jpg')
    if check:
        print(content)