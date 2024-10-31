import cv2
import dlib
import numpy as np

class FaceDetector:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)

    def detect_and_display(self):
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)

            if len(faces) != 0:
                for k, d in enumerate(faces):
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))

                    shape = self.predictor(im_rd, d)
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), -1, 8)

                    # 情绪判断逻辑部分
                    emotions = self.detect_emotion(shape)
                    cv2.putText(im_rd, emotions, (d.left(), d.top()-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            else:
                cv2.putText(im_rd, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Face Detector", im_rd)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def detect_emotion(self, shape):
        # 简单情绪判断逻辑
        mouth_width = shape.part(54).x - shape.part(48).x
        mouth_height = shape.part(66).y - shape.part(62).y
        brow_left = shape.part(21).y - shape.part(19).y
        brow_right = shape.part(22).y - shape.part(24).y
        brow_avg = (brow_left + brow_right) / 2

        if mouth_width > 50 and mouth_height > 20:
            return "Happy"
        elif brow_avg > 5:
            return "Angry"
        elif mouth_height > 20:
            return "Surprised"
        else:
            return "Neutral"

if __name__ == "__main__":
    face_detector = FaceDetector()
    face_detector.detect_and_display()
