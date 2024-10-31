import cv2
import dlib
from deepface import DeepFace


class FaceDetector:

    def __init__(self):
        # 使用特征提取器 get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib 的68点模型，使用官方训练好的特征预测器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 建cv2摄像头对象
        self.cap = cv2.VideoCapture(0)
        # 设置视频参数
        self.cap.set(3, 480)
        # 初始化帧计数器
        self.frame_count = 0

        # 存储分析后的人脸属性
        self.last_age = None
        self.last_emotion = None
        self.last_gender = None

    def detect_and_display(self):
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            if not flag:
                print("Failed to grab frame")
                break

            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            # 使用人脸检测器检测每一帧图像中的人脸
            faces = self.detector(img_gray, 0)

            if len(faces) != 0:
                # 对每个人脸都标出68个特征点
                for k, d in enumerate(faces):
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                    shape = self.predictor(im_rd, d)
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), -1)

                    # 每10帧进行一次深度分析
                    if self.frame_count % 10 == 0:
                        # 保存当前帧图像
                        cv2.imwrite("temp.jpg", im_rd)
                        try:
                            face_attributes = DeepFace.analyze(img_path="temp.jpg", actions=['age', 'emotion', 'gender'], enforce_detection=False)
                            # 处理分析结果
                            self.process_face_attributes(face_attributes)
                        except Exception as e:
                            print("Error in DeepFace analysis:", e)

                    # 显示分析的属性信息
                    self.display_attributes(im_rd, d)

            # 帧数累加
            self.frame_count += 1
            # 显示窗口
            cv2.imshow("Face Detector", im_rd)
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放摄像头
        self.cap.release()
        # 关闭所有窗口
        cv2.destroyAllWindows()

    # 存储分析后的人脸属性
    def process_face_attributes(self, face_attributes):
        if isinstance(face_attributes, list) and face_attributes:
            face_attributes = face_attributes[0]
        self.last_age = face_attributes.get('age')
        self.last_emotion = face_attributes.get('dominant_emotion')
        self.last_gender = face_attributes.get('gender')

    # 显示人脸属性
    def display_attributes(self, im_rd, d):
        if self.last_age and self.last_emotion and self.last_gender:
            cv2.putText(im_rd, f"Age: {self.last_age}", (d.left(), d.top() - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(im_rd, f"Emotion: {self.last_emotion}", (d.left(), d.top() - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(im_rd, f"Gender: {self.last_gender}", (d.left(), d.top() - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


if __name__ == "__main__":
    face_detector = FaceDetector()
    face_detector.detect_and_display()
