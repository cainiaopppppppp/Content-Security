import cv2
import dlib
from deepface import DeepFace

class FaceDetector:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)
        self.frame_count = 0
        self.last_age = None  # 用于保存上次检测的年龄
        self.last_emotion = None  # 用于保存上次检测的情绪

    def detect_and_display(self):
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            if not flag:
                print("Failed to grab frame")
                break
            k = cv2.waitKey(1)
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)

            if len(faces) != 0:
                for k, d in enumerate(faces):
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))

                    if self.frame_count % 10 == 0:
                        # Temporarily save the image to a file because DeepFace.analyze might not accept numpy array directly
                        cv2.imwrite("temp.jpg", im_rd)
                        # 使用DeepFace分析人脸属性
                        try:
                            face_attributes = DeepFace.analyze(img_path="temp.jpg", actions=['age', 'emotion'], enforce_detection=False)
                            # Check the type of face_attributes and handle accordingly
                            if isinstance(face_attributes, list):
                                face_attributes = face_attributes[0]  # Assuming the first face in the list
                            self.last_age = face_attributes['age']
                            self.last_emotion = face_attributes['dominant_emotion']
                        except Exception as e:
                            print("Error in DeepFace analysis:", e)

                    # Display the last known age and emotion
                    if self.last_age and self.last_emotion:
                        cv2.putText(im_rd, f"Age: {self.last_age}, Emotion: {self.last_emotion}", (d.left(), d.top() - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            self.frame_count += 1

            cv2.imshow("Face Detector", im_rd)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_detector = FaceDetector()
    face_detector.detect_and_display()
