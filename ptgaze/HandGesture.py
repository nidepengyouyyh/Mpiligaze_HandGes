import cv2
import torch
import mediapipe as mp
import numpy as np
import time

from method.basic_method import compute_angle
from method.basic_method import detect_all_finger_state, check_for_index_only
from method.extend_finger import line_from_points, point_line_distance, extend_line, does_line_intersect_box

class GestureObjectDetection:
    def __init__(self, camera_index=0, desktop_classes=None, object_model = None):
        self.i = 0
        # 上次检测到手势的时间
        self.last_gesture_time = None
        # 存储三秒前的指向直线
        self.last_line = None
        # 加载YOLOv5模型（默认使用yolov5n）
        self.model = object_model

        # 配置MediaPipe手势识别
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 限制识别手的数量
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)

        # 手指颜色
        self.finger_color = {
            'THUMB': (255, 0, 0),  # 蓝色
            'INDEX': (0, 255, 0),  # 绿色
            'MIDDLE': (0, 0, 255),  # 红色
            'RING': (255, 255, 0),  # 青色
            'PINKY': (255, 0, 255)  # 洋红色
        }

        # 相机初始化
        self.cap = cv2.VideoCapture(camera_index)

    def get_box_center(self, x1, y1, x2, y2):
        """ 获取YOLO框的中心点 """
        return (x1 + x2) / 2, (y1 + y2) / 2

    def draw_finger_custom_color(self, image, hand_landmarks, color_map):
        """ 绘制手指并自定义颜色 """
        finger_tips = {
            'THUMB': [1, 2, 3, 4],
            'INDEX': [5, 6, 7, 8],
            'MIDDLE': [9, 10, 11, 12],
            'RING': [13, 14, 15, 16],
            'PINKY': [17, 18, 19, 20]
        }

        for finger_name, landmark_indices in finger_tips.items():
            for idx in landmark_indices:
                lm = hand_landmarks.landmark[idx]
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 10, color_map[finger_name], -1)

    def hand_points(self, hand_landmarks):
        """ 获取手部关键点坐标 """
        points = {
            'point0': (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y),  # Wrist
            'point1': (hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y),  # Thumb CMC
            'point2': (hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y),  # Thumb MCP
            'point3': (hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y),  # Thumb IP
            'point4': (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y),  # Thumb Tip
            'point5': (hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y),  # Index MCP
            'point6': (hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y),  # Index PIP
            'point7': (hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y),  # Index DIP
            'point8': (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y),  # Index Tip
            'point9': (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y),  # Middle MCP
            'point10': (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y),  # Middle PIP
            'point11': (hand_landmarks.landmark[11].x, hand_landmarks.landmark[11].y),  # Middle DIP
            'point12': (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y),  # Middle Tip
            'point13': (hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y),  # Ring MCP
            'point14': (hand_landmarks.landmark[14].x, hand_landmarks.landmark[14].y),  # Ring PIP
            'point15': (hand_landmarks.landmark[15].x, hand_landmarks.landmark[15].y),  # Ring DIP
            'point16': (hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y),  # Ring Tip
            'point17': (hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y),  # Pinky MCP
            'point18': (hand_landmarks.landmark[18].x, hand_landmarks.landmark[18].y),  # Pinky PIP
            'point19': (hand_landmarks.landmark[19].x, hand_landmarks.landmark[19].y),  # Pinky DIP
            'point20': (hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y)  # Pinky Tip
        }
        return points

    def process_frame(self, frame, filtered_detections):
        """ 处理每一帧 """
        # MediaPipe 手势识别
        gudge=0#初始化判断是否有物体指向
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = self.hands.process(frame_rgb)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.draw_finger_custom_color(frame_bgr, hand_landmarks, self.finger_color)
                self.mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                all_points = self.hand_points(hand_landmarks)

                bend_states, straighten_states = detect_all_finger_state(all_points)

                # 检查是否只伸出食指
                if check_for_index_only(straighten_states, bend_states):
                    print("提示：开始指向")
                    # 生成食指所在的直线方程
                    start_point = (
                    int(all_points['point5'][0] * frame.shape[1]), int(all_points['point5'][1] * frame.shape[0]))
                    end_point = (
                    int(all_points['point8'][0] * frame.shape[1]), int(all_points['point8'][1] * frame.shape[0]))

                    line = line_from_points(start_point, end_point)

                    # 延长食指所在的直线
                    (x1, y1), (x2, y2) = extend_line(line, frame.shape[0])
                    # 绘制食指所在的直线
                    cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色直线

                    # 判断食指直线是否穿过物体框，并计算最短距离
                    for detection in filtered_detections:
                        box_x1, box_y1, box_x2, box_y2, _, _ = detection
                        box_center = self.get_box_center(box_x1, box_y1, box_x2, box_y2)

                        min_distance = point_line_distance(line, box_center)
                        print(min_distance)
                        line2 = ((x1, y1), (x2, y2))
                        if min_distance < 150:  # 设置一个阈值来判断是否足够接近
                            if does_line_intersect_box(line2, (box_x1, box_y1, box_x2, box_y2)):
                                if self.i==0:
                                    self.last_line =((x1, y1), (x2, y2))
                                    self.last_gesture_time=time.time()
                                    self.i=1
                                print("食指指向了物体:", self.model.names[int(detection[5])])
                                current_line=((x1, y1), (x2, y2))
                                # 存储当前的指向直线

                                # 检查是否有2秒前的指向直线
                                if self.last_line is not None and (time.time() - self.last_gesture_time) >= 2:
                                    # 计算两条直线的角度差
                                    (xl1, yl1), (xl2, yl2) = self.last_line
                                    (xf3, yf3), (xf4, yf4) = current_line
                                    angle_diff = compute_angle(xl1, yl1,xl2, yl2,xf3, yf3,xf4, yf4)* 180 / np.pi
                                    if angle_diff <5:
                                        # 更新2秒前的指向直线
                                        self.last_line = current_line
                                        self.i = 0
                                        gudge = 1
                                    else:
                                        gudge = 0

                            else:
                                print("未穿过物体框")

                else:
                    if self.last_gesture_time is None or (time.time() - self.last_gesture_time) > 3:
                        print("无手势")
        # frame_bgr = cv2.flip(frame_bgr,1)
        return frame_bgr,gudge

    def start(self):
        """ 开始视频处理 """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # 处理每一帧
            processed_frame = self.process_frame(frame,)

            # 显示处理后的帧
            cv2.imshow('MediaPipe Hands + YOLOv5', processed_frame)

            # 按下ESC键退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    object_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    detection = GestureObjectDetection(object_model = object_model)
    detection.start()
