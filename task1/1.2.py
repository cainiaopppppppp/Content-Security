import argparse
import cv2.dnn
import numpy as np


# 预定义的COCO数据集的80个类别
CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
           7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
           13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
           21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
           28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
           41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
           50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
           58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
           66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
           73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# 为每个类别分配随机颜色
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 绘制边界框和类别标签
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # 准备标签文本，包括类别名称和置信度
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    # 获取对应类别的颜色
    color = colors[class_id]
    # 在图像上绘制边界框
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # 在图像上绘制标签文本
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main(onnx_model, input_image):
    # 从ONNX文件加载模型
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    # 读取输入图像
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    length = max((height, width))
    # 创建方形图像，以适应YOLOv8模型的输入需求
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    # 计算缩放比例，以还原边界框坐标到原图尺寸
    scale = length / 640
    # 为模型准备输入blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    # 执行模型推理，获取输出
    outputs = model.forward() # output: 1 X 8400 x 84
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    # 初始化列表，用于存储检测结果
    boxes = []
    scores = []
    class_ids = []
    # 解析模型输出，筛选出置信度高于阈值的检测结果
    for i in range(rows):
        classes_scores = outputs[0][i][4:]  # 提取当前行的类别得分
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:  # 置信度阈值
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)  # 添加边界框
            scores.append(maxScore)  # 添加置信度
            class_ids.append(maxClassIndex)  # 添加类别索引
        # 使用非极大值抑制（NMS）进一步筛选边界框
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    # 在原图上绘制最终筛选出的边界框和类别标签
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

    # 显示结果图像
    cv2.imshow('image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='yolov8n.onnx', help='Input your onnx models.')
    parser.add_argument('--img', default=str('D:\Content_Secu\INRIAPerson\INRIAPerson\Train\person\crop_000011.png'), help='Path to input image.')
    args = parser.parse_args()
    main(args.model, args.img)

