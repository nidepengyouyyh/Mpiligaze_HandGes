import threading
import time
from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np
import os

class VisionProcessor:
    def __init__(self, model_path: str, show:bool=False):
        self.model = YOLO(model_path)
        self.last_update = time.time()
        self.detected_objects = []
        self.lock = threading.Lock()  # 线程安全锁
        self.show = show
        self.summary = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.current_instruction = None  # 存储原始指令

    def start_detection_loop(self, camera_index=0):
        """启动独立检测线程"""
        def detection_loop():
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while True:
                ret, frame = cap.read()
                results = self.model(frame, verbose=False)
                with self.frame_lock:
                    self.current_frame = frame.copy()  # 保存最新帧

                # YOLO检测...
                with self.lock:
                    self.detected_objects = self._parse_results(results)
                    self._check_confidence()  # 新增置信度检查

        threading.Thread(target=detection_loop, daemon=True).start()

    def _check_confidence(self):
        if not self.summary:
            return

        # 遍历每个summary目标（如red cube）
        for target in self.summary:
            # 将目标关键词转为小写并拆分为单词集合
            # 示例：'red cube' → {'red', 'cube'}
            target_words = set(target.lower().split())

            # 筛选出符合要求的检测对象
            # obj结构示例：{'class': 'cube', 'confidence': 0.75, ...}
            relevant_confs = []
            for obj in self.detected_objects:
                # 获取当前检测对象的类别单词集合
                # 示例：obj['class'] = 'cube' → {'cube'}
                obj_class_words = set(obj['class'].lower().split())

                # 检查是否为子集关系
                if obj_class_words.issubset(target_words):
                    relevant_confs.append(obj['confidence'])

            # 计算最高置信度（若无匹配项则为0）
            max_conf = max(relevant_confs, default=0.0)

            # 触发条件判断
            if max_conf < 0.6:
                print(f"触发VLM：目标 '{target}' 最高置信度 {max_conf:.2f}")
                self._capture_and_process()
                break  # 发现一个低置信目标即终止检查

    def _capture_and_process(self):
        """保存当前帧并调用VLM"""
        timestamp = int(time.time())
        filename = os.path.join(self.temp_dir, f"capture_{timestamp}.jpg")

        with self.frame_lock:
            if self.current_frame is not None:
                cv2.imwrite(filename, self.current_frame)

        # 异步调用VLM（避免阻塞检测线程）
        threading.Thread(
            target=self._call_vlm,
            args=(filename, self.current_instruction)
        ).start()

    @staticmethod
    def _call_vlm(img_path: str, instruction: str) -> list:
        """VLM输出格式转换器
        返回与YOLO相同的结构：[{'class':str, 'confidence':float, 'position':tuple}]
        """
        from vlm import qwen_vl_max_last
        import cv2

        # 获取图片尺寸用于坐标归一化
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # 调用VML接口
        success, vlm_data = qwen_vl_max_last(instruction, img_path)
        if not success:
            return []

        # 格式转换器
        def convert_vlm_item(name_key, xyxy_key):
            """将VLM单条数据转为YOLO格式"""
            x1, y1 = vlm_data[xyxy_key][0]
            x2, y2 = vlm_data[xyxy_key][1]
            return {
                'class': vlm_data[name_key],  # 与YOLO类别名保持一致
                'confidence': 1.0,  # VLM默认置信度
                'position': (
                    (x1 + x2) / 2 / img_w,  # 归一化X坐标
                    (y1 + y2) / 2 / img_h  # 归一化Y坐标
                )
            }

        # 返回标准化结果（示例最多支持两个对象）
        return [
            convert_vlm_item('start', 'start_xyxy'),
            convert_vlm_item('end', 'end_xyxy')
        ]

    @staticmethod
    def _parse_results(results):
        """解析检测结果"""
        return [{
            'class': results[0].names[int(cls_id)],
            'confidence': float(conf),
            'position': ((x1+x2)/2/640, (y1+y2)/2/480)  # 归一化坐标
        } for x1, y1, x2, y2, conf, cls_id in results[0].boxes.data.cpu().numpy()]

    def get_visual_context(self) -> str:
        """生成视觉提示词"""
        with self.lock:
            objects = self.detected_objects
            if not objects:
                return "当前视野内未检测到任何物体"

            # 按类别聚合信息
            counter = defaultdict(list)
            for obj in objects:
                counter[obj['class']].append(obj)

            descriptions = []
            for cls, items in counter.items():
                pos_desc = "、".join([self._get_position_desc(obj['position']) for obj in items[:3]])
                desc = f"{cls}（{len(items)}个，主要分布在{pos_desc}）"
                descriptions.append(desc)
            return "视觉感知结果：\n- " + "\n- ".join(descriptions)

    @staticmethod
    def _get_position_desc(pos) -> str:
        """将坐标转换为自然语言描述"""
        x, y = pos
        vertical = "上方" if y < 0.3 else "下方" if y > 0.7 else "中间区域"
        horizontal = "左侧" if x < 0.3 else "右侧" if x > 0.7 else "中部"
        return f"{horizontal}{vertical}"

    def show_it(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低分辨率
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 执行推理
            results = self.model(frame, verbose=False)

            # 提取检测信息
            if results[0].boxes.shape[0] > 0:  # 检测到物体时
                boxes = results[0].boxes.xyxy.cpu().numpy()  # 转换为numpy数组
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                # 遍历所有检测框
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    hue = int(180 * i / max(len(boxes), 1))  # 防止除以零
                    hsv_color = np.uint8([[[hue, 255, 255]]])
                    color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    color = tuple(map(int, color))
                    # 计算中心坐标
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # 获取类别信息
                    class_name = results[0].names[class_ids[i]]

                    # 绘制图形
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame,
                                f"{class_name} {confidences[i]:.2f} {(center_x, center_y)}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (10, 255, 20), 2)

            cv2.imshow('YOLO Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def update_summary(self, summary: list, instruction: str):
        """示例输入:
        summary = ['red cube', 'house sketch']
        instruction = "请帮我把红色方块放在房子简笔画上"
        """
        self.summary = summary
        self.current_instruction = instruction

if __name__ == '__main__':
    vision = VisionProcessor("yl/yolo11s.pt", True)
    vision.show_it()