from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import JsonResponse
from .models import Visitor, VisitHistory
from .serializers import VisitorSerializer, VisitHistorySerializer
import base64
import numpy as np
import cv2
from PIL import Image
import io
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from .test_model_2 import test
import traceback
from .mtcnn3 import MTCNN3
from keras.models import load_model

# 使用numpy读取中文路径的图像
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


# 添加用户模块
class VisitorView(APIView):
    def post(self, request):
        serializer = VisitorSerializer(data=request.data)
        if serializer.is_valid():
            visitor = serializer.save()
            face_image_path = visitor.face_image.path
            image = cv_imread(face_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_detector.setInputSize((image.shape[1], image.shape[0]))
            _, faces = face_detector.detect(image)

            if faces is None or len(faces) == 0:
                return Response({'error': '未检测到人脸。'}, status=status.HTTP_400_BAD_REQUEST)

            # 人脸特征提取
            face_aligned = face_recognizer.alignCrop(image, faces[0])
            face_feature = face_recognizer.feature(face_aligned)

            # 展平并保存特征
            face_feature_flat = face_feature.flatten() if len(face_feature.shape) > 1 else face_feature
            visitor.face_features = face_feature_flat.tobytes()
            visitor.save()

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 签到记录
class VisitHistoryView(generics.ListAPIView):
    queryset = VisitHistory.objects.all().order_by('-timestamp')  # 按时间降序排列
    serializer_class = VisitHistorySerializer


# 初始化模型1
def load_mobilenet_model():
    model_path = './model/model1/mobilenetv2-best.hdf5'
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model.output
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights(model_path)
    return model

liveness_model = load_mobilenet_model()

# OpenCV 面部检测和识别
face_detector = cv2.FaceDetectorYN.create('./model/model1/face_detection.onnx', '', (320, 320), 0.9, 0.3, 5000)
face_recognizer = cv2.FaceRecognizerSF.create('./model/model1/face_recognition.onnx', '')

# 初始化模型2
# 使用 MTCNN 和 MiniFASNet 初始化
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

model_dir = "./model/model2/anti_spoof_models"

def model2_predict(image_np):
    """利用第二个模型进行活体检测"""
    frame_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(frame_rgb)

    if boxes is not None and len(boxes) > 0:
        for box, prob in zip(boxes, probs):
            print(prob)
            if prob > 0.8:
                face = frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                face = cv2.resize(face, (128, 128))

                spoof_label = test(face, model_dir, 0)
                print()
                return 0 if spoof_label == 0 else 1
    return 1


# 初始化模型3
model3_path = "./model/model3/fas.h5"
model3 = load_model(model3_path)
mtcnn_model3 = MTCNN3(
    './model/model3/mtcnn.pb',
    min_size=40,  # 适当降低最小尺寸
    factor=0.7,   # 调整缩放因子
    thresholds=[0.6, 0.7, 0.7]  # 调整检测阈值
)


def model3_predict(image_np):
    """利用模型3进行检测，将整张图像调整为模型输入大小"""
    image_resized = cv2.resize(image_np, (224, 224))
    print(f"Image resized shape: {image_resized.shape}")

    image_standardized = (image_resized - 127.5) / 127.5
    input_data = np.expand_dims(image_standardized, axis=0)

    score = model3.predict(input_data)[0]
    print(f"Score: {score}")

    return "Real" if score > 0.1 else "Fake"


class ValidateVisitorAPIView(APIView):
    def post(self, request):
        try:
            image_data = request.data.get('image', '')
            student_id = request.data.get('student_id', '')
            model_choice = request.data.get('model', 'model1')

            if not image_data or not student_id:
                return JsonResponse({'error': '缺少图像或学生ID。'}, status=400)

            # 解码并转换图像
            image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image))
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # 根据所选模型执行检测
            if model_choice == 'model1':
                resized_image = cv2.resize(image_np, (224, 224))
                img_array = resized_image.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = liveness_model.predict(img_array)[0][0]

                print(prediction)
                if prediction <= 0.4:
                    return JsonResponse({'result': '拒绝，不是真人。'}, status=403)

            elif model_choice == 'model2':
                prediction = model2_predict(image_np)
                print(prediction)
                if prediction == 1:
                    return JsonResponse({'result': '拒绝，不是真人。'}, status=403)

            elif model_choice == 'model3':
                result = model3_predict(image_np)
                if result == "Fake" or result == "No Face Detected":
                    return JsonResponse({'result': f'拒绝，{result}。'}, status=403)

            else:
                return JsonResponse({'error': '无效的模型选择。'}, status=400)

            # 通用人脸检测与验证逻辑
            face_detector.setInputSize((image_np.shape[1], image_np.shape[0]))
            _, faces = face_detector.detect(image_np)

            if faces is None or len(faces) == 0:
                return JsonResponse({'result': '未检测到人脸。'}, status=404)

            face_aligned = face_recognizer.alignCrop(image_np, faces[0])
            face_feature = face_recognizer.feature(face_aligned)

            try:
                visitor = Visitor.objects.get(student_id=student_id)
                stored_face_feature = np.frombuffer(visitor.face_features, dtype=np.float32)
                face_feature_flat = face_feature.flatten() if len(face_feature.shape) > 1 else face_feature
                stored_face_feature = stored_face_feature.reshape(face_feature_flat.shape)

                similarity = np.dot(stored_face_feature, face_feature_flat) / (
                    np.linalg.norm(stored_face_feature) * np.linalg.norm(face_feature_flat))

                print(similarity)
                if similarity < 0.45:
                    return JsonResponse({'result': '面部与数据库不匹配。'}, status=403)

                VisitHistory.objects.create(visitor=visitor)
                return JsonResponse({'result': '成功认证。', 'visitor': VisitorSerializer(visitor).data})

            except Visitor.DoesNotExist:
                return JsonResponse({'result': '数据库中未找到访客。'}, status=404)

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"ValidateVisitorAPIView 错误：{e}\n{error_trace}")
            return JsonResponse({'error': f'错误：{str(e)}'}, status=500)
