import numpy as np
import cv2 as cv
import pathlib, os


# 灰度图转换
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


# Canny边缘检测
def canny(image, low_threshold, high_threshold):
    return cv.Canny(image, low_threshold, high_threshold)


# 高斯滤波
def gaussian_blur(image, kernel_size):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)


# 生成感兴趣区域即Mask掩模
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)  # 生成图像大小一致的zeros矩

    # 填充顶点vertices中间区域
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # 填充函数
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


# 原图像与车道线图像按照a:b比例融合
def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv.addWeighted(initial_img, a, img, b, c)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    slope_min = .15  # 斜率低阈值
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
            slope = fit[0]  # 斜率
            if slope_min < np.absolute(slope):
                cv.line(image, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # rho：线段以像素为单位的距离精度
    # theta : 像素以弧度为单位的角度精度(np.pi/180较为合适)
    # threshold : 霍夫平面累加的阈值
    # minLineLength : 线段最小长度(像素级)
    # maxLineGap : 最大允许断裂长度
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)
    return lines


def process_image(image):
    rho = 1  # 霍夫像素单位
    theta = np.pi / 180  # 霍夫角度移动步长
    hof_threshold = 100  # 霍夫平面累加阈值threshold
    min_line_len = 100  # 线段最小长度
    max_line_gap = 8  # 最大允许断裂长度

    kernel_size = 5  # 高斯滤波器大小size
    canny_low_threshold = 75  # canny边缘检测低阈值
    canny_high_threshold = canny_low_threshold * 3  # canny边缘检测高阈值

    alpha = 0.8  # 原图像权重
    beta = 1.  # 车道线图像权重
    lambda_ = 0.

    imshape = image.shape  # 获取图像大小

    # 灰度图转换
    gray = grayscale(image)

    # 高斯滤波
    blur_gray = gaussian_blur(gray, kernel_size)

    # Canny边缘检测
    edge_image = canny(blur_gray, canny_low_threshold, canny_high_threshold)

    # 生成Mask掩模
    vertices = np.array([[(0, imshape[0]), (0, imshape[0] / 3),
                          (imshape[1], imshape[0] / 3), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edge_image, vertices)
    # 基于霍夫变换的直线检测
    lines = hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)
    line_image = np.zeros_like(image)

    # 绘制车道线线段
    draw_lines(line_image, lines, thickness=10)

    # 图像融合
    lines_edges = weighted_img(image, line_image, alpha, beta, lambda_)
    return lines_edges


if __name__ == '__main__':
    # img_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/img/1492626274615008344/1.jpg"
    # image = cv.imread(img_path)
    # line_image = process_image(image)
    # cv.imwrite("/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/res/result.jpg", line_image)

    # 定义文件夹路径
    folder_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/img/1492626127172745520_0"
    output_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/res"
    # 获取文件夹路径下的所有的文件名
    file_names = [file for file in pathlib.Path(folder_path).iterdir()]
    for file_name in file_names:
        img_path = os.path.join(folder_path, file_name.name)
        image = cv.imread(img_path)
        line_image = process_image(image)
        cv.imwrite(os.path.join(output_path, file_name.name), line_image)
