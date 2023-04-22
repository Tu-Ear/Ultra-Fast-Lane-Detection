import cv2
import numpy as np
from PIL import Image, ImageEnhance


def adjust_brightness(img, brightness_factor):
    """
    调整图片的亮度
    :param img: PIL.Image对象
    :param brightness_factor: 亮度调整系数，值越大，图像越亮
    :return: PIL.Image对象
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness_factor)


def adjust_contrast(img, contrast_factor):
    """
    调整图片的对比度
    :param img: PIL.Image对象
    :param contrast_factor: 对比度调整系数，值越大，对比度越高
    :return: PIL.Image对象
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(contrast_factor)


def adjust_saturation(img, saturation_factor):
    """
    调整图片的饱和度
    :param img: PIL.Image对象
    :param saturation_factor: 饱和度调整系数，值越大，饱和度越高
    :return: PIL.Image对象
    """
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(saturation_factor)


def transform_night_to_day(image_path):
    bright_factor = 1.2  # 亮度系数
    contrast_factor = 1.2  # 对比度系数
    saturation_factor = 1.2  # 饱和度系数
    image = Image.open(image_path).convert('RGB')
    # 调整图像的亮度、对比度、饱和度
    image = adjust_brightness(image, bright_factor)
    image = adjust_contrast(image, contrast_factor)
    image = adjust_saturation(image, saturation_factor)
    return image

def enhance_lane_lines(image_path):
    # 加载输入图像并将其转换为YUV颜色空间
    img = cv2.imread(image_path)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 分离颜色通道
    y, u, v = cv2.split(yuv)

    # 对亮度通道进行直方图均衡化
    y_eq = cv2.equalizeHist(y)

    # 对亮度通道进行高斯滤波
    blur = cv2.GaussianBlur(y_eq, (5, 5), 0)

    # 进行边缘检测，将车道线轮廓强调出来
    edges = cv2.Canny(blur, 50, 150)

    # 对图像进行二值化处理，将车道线区域分割出来
    ret, thresh = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)

    # 对二值化图像进行膨胀操作，强化车道线
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=2)

    # 将强化后的车道线图像与原始图像合并
    enhanced_image = cv2.merge((y_eq, u, v))
    result = cv2.bitwise_and(enhanced_image, enhanced_image, mask=dilation)

    # 将图像转换回RGB颜色空间
    result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)

    # 返回强化后的车道线图像
    return result

def enhance_edges(image_path):
    # 加载输入图像
    img = cv2.imread(image_path)

    # 对图像进行高斯模糊处理，以减少噪声
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 对图像进行边缘检测，强化边缘信息
    edges = cv2.Canny(blur, 100, 200)

    # 将边缘图像与原始图像合并，保留颜色信息
    result = cv2.bitwise_and(img, img, mask=edges)

    # 返回强化后的边缘图像
    return result




# 调用enhance_lane_lines()函数并显示结果
if __name__ == '__main__':
        image_path = "/data/ldp/zjf/infer/00270.jpg"  # 输入的RGB图像文件路径
        result_image = enhance_edges(image_path)
        cv2.imwrite('/data/ldp/zjf/infer/output.jpg', result_image)

# img = transform_night_to_day("/data/ldp/zjf/infer/00270.jpg")
    # img = np.array(img)
    # cv2.imwrite('/data/ldp/zjf/infer/output.jpg', img)

    # 读取图像
    # img = cv2.imread('/data/ldp/zjf/infer/00810.jpg')

    # # 转化为灰度图像
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # 对图像进行增强，提高对比度和亮度
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)
    #
    # # 对图像进行调整，增加亮度和清晰度
    # dst = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 输出处理后的图像
    # cv2.imwrite('/data/ldp/zjf/infer/output.jpg', dst)
