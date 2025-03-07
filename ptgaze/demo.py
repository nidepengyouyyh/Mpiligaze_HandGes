import datetime
import logging
import pathlib
from typing import Optional
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
from omegaconf import DictConfig
from method.basic_method import compute_angle
from common import Face, FacePartsName, Visualizer
from gaze_estimator import GazeEstimator
from pt_utils import get_3d_face_model
from HandGesture import GestureObjectDetection
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from method.extend_finger import line_from_points, point_line_distance, extend_line, does_line_intersect_box

import sys
sys.path.append('/home/nidepengyouyyh/study/Codes/Mpiligaze_handGesture')
from agent.llm_demo import listen_vlm

class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.eye_pt0 = None
        self.eye_pt1 = None
        self.mode = None
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

        self.object_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
        self.gesture_detector = GestureObjectDetection(object_model = self.object_model)
        # 默认桌面物品类别ID
        self.desktop_classes = [39, 40, 41, 42, 44, 45, 46, 47, 48, 67, 31, 73]
        self.speech = listen_vlm()
        # 上次检测到手势的时间
        self.last_gesture_time = None
        # 存储三秒前的指向直线
        self.last_line = None
        self.i = 0
        self.gudge = 0

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    @staticmethod
    def frame_to_base64(frame):
        """将OpenCV帧转换为带前缀的Base64编码字符串（与_encode_image格式对齐）"""
        # 转换颜色空间 BGR转RGB（OpenCV默认BGR，PIL需要RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # 生成JPEG字节数据
        output_buffer = BytesIO()
        img.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()

        # 生成带前缀的base64字符串（与_encode_image完全一致）
        return 'data:image/jpeg;base64,' + base64.b64encode(byte_data).decode('utf-8')

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def yolo_detect(self, frame):
        # YOLOv5 物体检测
        results_yolo = self.object_model(frame)
        detections = results_yolo.xyxy[0].cpu().numpy()  # 获取检测框的坐标和类别

        # 过滤掉不在指定类别内的检测结果
        filtered_detections = [detection for detection in detections if int(detection[5]) in self.desktop_classes]

        # 绘制过滤后的检测框
        for detection in filtered_detections:
            x1, y1, x2, y2, conf, class_id = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{self.object_model.names[int(class_id)]} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return filtered_detections

    def get_box_center(self, x1, y1, x2, y2):
        """ 获取YOLO框的中心点 """
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _run_on_video(self) -> None:
        frame_count = 0
        check = True  # 计数器，用于追踪帧的数量

        try:
            while True:
                if self.mode is None:
                    self.mode = 'normal'
                if self.config.demo.display_on_screen:
                    self._wait_key()
                    if self.stop:
                        break

                ok, frame = self.cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)

                img = self.frame_to_base64(frame)
                self.speech.add_img_to_queue(img)  # 将图像添加到语音识别队列

                recognized_mode = self.speech.get_mode()
                # print(recognized_mode)
                if recognized_mode != self.mode and recognized_mode in ['hand', 'eye', 'normal', 'exit']:
                    print(f">> 切换到 {recognized_mode} 模式")
                    if recognized_mode =='exit' and self.mode != 'normal':
                        self.mode = 'normal'
                        self.speech.set_mode('normal')
                    elif recognized_mode =='exit' and self.mode == 'normal':
                        return 0
                    else:
                        self.mode = recognized_mode

                # print(self.mode)
                # 调整帧的大小
                frame_resized = cv2.resize(frame, (640, 480))
                filtered_detections = self.yolo_detect(frame_resized)

                # 移除了直接处理音频的部分，改为依赖后台线程的结果

                if frame_count % 2 == 0:
                    if self.mode == 'hand':
                        check = False
                        # 处理手势识别
                        self.visualizer.image, result = self.gesture_detector.process_frame(frame_resized,
                                                                                                filtered_detections)
                        if result == 1:
                            print("已检测到物体，退出手部识别程序")
                            self.mode = 'normal'
                    elif self.mode == 'eye':
                        check = False
                        # 处理眼部注视方向等
                        self._process_image(frame_resized, filtered_detections)
                        if self.gudge == 1:
                            print("已检测到物体，退出眼动识别程序")
                            self.mode = 'normal'
                            self.gudge = 0
                    elif self.mode == 'normal':
                        self.visualizer.image = frame_resized

                # 增加计数器，直到达到2重新开始
                frame_count += 1

                # 显示处理后的图像
                if self.config.demo.display_on_screen:
                    cv2.imshow('frame', self.visualizer.image)

                key = cv2.waitKey(1)
                if key == ord('r'):
                    check = not check

            self.cap.release()
            if self.writer:
                self.writer.release()

        except KeyboardInterrupt:
            print(">> 用户主动终止")
        except Exception as e:
            print(f">> 发生异常: {str(e)}")
        finally:
            print(">> 正在清理资源...")
            self.speech._cleanup_audio_resources()  # 如果这是必要的，可以保留
            print(">> 服务已安全关闭")

    def _process_image(self, image, filtered_detections) -> None:
        gudge = 0
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        # 判断食指直线是否穿过物体框，并计算最短距离
        line_3d = line_from_points(self.eye_pt0, self.eye_pt1)

        for detection in filtered_detections:
            box_x1, box_y1, box_x2, box_y2, _, _ = detection
            box_center = self.get_box_center(box_x1, box_y1, box_x2, box_y2)

            min_distance = point_line_distance(line_3d, box_center)
            print(min_distance)
            if min_distance < 100:  # 设置一个阈值来判断是否足够接近
                if does_line_intersect_box((self.eye_pt0, self.eye_pt1), (box_x1, box_y1, box_x2, box_y2)):
                    if self.i == 0:
                        self.last_line = (self.eye_pt0, self.eye_pt1)
                        self.last_gesture_time = time.time()
                        self.i = 1
                    print("用户在看物体:", self.object_model.names[int(detection[5])])
                    current_line = (self.eye_pt0, self.eye_pt1)
                    if self.last_line is not None and (time.time() - self.last_gesture_time) >= 2:
                        # 计算两条直线的角度差
                        (xl1, yl1), (xl2, yl2) = self.last_line
                        (xf3, yf3), (xf4, yf4) = current_line
                        angle_diff = (compute_angle(xl1, yl1, xl2, yl2, xf3, yf3, xf4, yf4) * 180 / np.pi)
                        if angle_diff < 5:
                            # 更新2秒前的指向直线
                            self.last_line = current_line
                            self.i = 0
                            self.gudge = 1
                        else:
                            self.gudge = 0
                else:
                    print("未穿过物体框")

        # if self.config.demo.use_camera:
        #     self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)
    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    #m模式切换，后续要改成语音输入
    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        # elif key == ord('k'):
        #     if self.mode == 'hand':
        #         self.mode = 'eye'  # 切换为 'eye'
        #     elif self.mode == 'eye':
        #         self.mode = 'hand'  # 切换回 'hand'
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            self.eye_pt0, self.eye_pt1 = self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector * 8)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')

        else:
            raise ValueError
