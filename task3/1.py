import cv2
import dlib

class FaceDetector():

    def __init__(self):
        # 使用特征提取器 get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib 的68点模型，使用官方训练好的特征预测器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 建cv2摄像头对象
        self.cap = cv2.VideoCapture(0)
        # 设置视频参数
        self.cap.set(3, 480)

    def detect_and_display(self):
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            k = cv2.waitKey(1)
            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸
            faces = self.detector(img_gray, 0)

            # 如果检测到人脸
            if (len(faces) != 0):
                # 对每个人脸都标出68个特征点

                for i in range(len(faces)):
                    for k, d in enumerate(faces):
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))

                        # 使用预测器得到68点数据的坐标
                        shape = self.predictor(im_rd, d)
                        # 圆圈显示每个特征点
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), -1, 8)
            else:
                cv2.putText(im_rd, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

            # 窗口显示
            cv2.imshow("Face Detector", im_rd)

            # 按下 q 键退出
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        # 释放摄像头
        self.cap.release()
        # 删除建立的窗口
        cv2.destroyAllWindows()


# main
if __name__ == "__main__":
    face_detector = FaceDetector()
    face_detector.detect_and_display()
