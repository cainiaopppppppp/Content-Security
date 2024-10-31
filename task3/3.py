import cv2
import dlib
import numpy

import sys

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
# 图像缩放比例
SCALE_FACTOR = 1
# 用于模糊掩码边界的参数
FEATHER_AMOUNT = 11

# 定义各个部分的特征点索引
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# 用于对齐图像的特征点
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# 用于覆盖的特征点组，使用凸包进行覆盖
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# 颜色校正的模糊度
COLOUR_CORRECT_BLUR_FRAC = 0.6

# 初始化dlib的人脸检测器和特征点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# 如果检测到多于一个人脸，抛出异常。
class TooManyFaces(Exception):
    pass

# 如果没有检测到人脸，抛出异常。
class NoFaces(Exception):
    pass


# 检测给定图像中的人脸，并提取68个面部特征点
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


#  在图像上标记面部特征点，包括在每个特征点位置绘制圆点和标号，用于调试或可视化特征点位置。
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


# 在图像上根据给定的点绘制凸包并填充指定颜色。这在生成面部区域掩码时用于定义要进行融合的面部区域。
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


# 生成面部掩码。根据特征点，为给定的面部区域（例如眼睛、鼻子、嘴巴等区域）生成一个掩码，用于之后的图像融合过程。
def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


# 根据两组特征点计算仿射变换矩阵，用于将一个人脸对齐到另一个人脸。
def transformation_from_points(points1, points2):

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


# 读取图像文件，调整其大小，并提取面部特征点。这是预处理步骤，为面部融合准备图像和数据。
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s


# 使用计算得到的仿射变换矩阵（M）来变换图像，使之与目标图像对齐。
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


# 颜色校正，以匹配两张融合图像的颜色。通过应用高斯模糊来平滑颜色差异，并调整颜色强度，使融合后的图像看起来更自然。
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))


# 读取图像及其特征
im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
im2, landmarks2 = read_im_and_landmarks(sys.argv[2])

# 计算从第一张图的特征点到第二张图特征点的仿射变换矩阵
M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

# 为第二张图生成面部掩码
mask = get_face_mask(im2, landmarks2)
# 将面部掩码应用仿射变换，使之与第一张图对齐
warped_mask = warp_im(mask, M, im1.shape)

# 合并第一张图和变形后的第二张图的面部掩码
combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

# 应用变换矩阵变形第二张图像，使之与第一张图像对齐
warped_im2 = warp_im(im2, M, im1.shape)
# 对变形后的第二张图像进行颜色校正，使其颜色与第一张图像更加吻合
warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

# 生成最终融合图像，通过掩码来混合两张图像的相应部分
output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

# 将最终融合后的图像保存到文件
cv2.imwrite('output.jpg', output_im)
