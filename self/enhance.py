# coding： utf-8
from PIL import Image, ImageFilter
import os, cv2
from PIL import ImageGrab
import numpy as np

#  源目录
input_path = '/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/img/1492626127172745520_0'
#  输出目录
output_path = '/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/enhance'


def imageEnhance(input_path, output_path):
    kernel_size = 5  # 高斯滤波器大小size

    # 获取输入文件夹中的所有文件/夹，并改变工作空间
    files = os.listdir(input_path)
    os.chdir(input_path)
    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in files:
        # 判断是否为文件，文件夹不操作
        if os.path.isfile(file):
            output_file = os.path.join(output_path, "New_" + file)
            # cv_img = cv2.imread(file)
            # cv2.imwrite(output_file, cv2.GaussianBlur(cv_img, (kernel_size, kernel_size), 0))

            img = Image.open(file)
            # 将图片缩放为96*96大小
            # img = img.resize((96, 96), Image.ANTIALIAS)
            # 边缘增强
            enhance = img.filter(ImageFilter.EDGE_ENHANCE)
            # # 找到边缘
            # edges = img.filter(ImageFilter.FIND_EDGES)
            # # 浮雕
            # img.filter(ImageFilter.EMBOSS)
            # # 轮廓
            # img.filter(ImageFilter.CONTOUR)
            # # 平滑
            # smooth = img.filter(ImageFilter.SMOOTH)
            # 锐化
            sharp = img.filter(ImageFilter.SHARPEN)

            # # 细节
            # img.filter(ImageFilter.DETAIL)
            enhance.save(output_file)

            # cv_img = cv2.imread(file)

            # 直方图均衡化
            # hist, bins = cv2.getHistcounts(cv_img, cv2.CV_8UC1)
            # max_val = np.max(hist)
            # min_val = np.min(hist)
            # step = max_val / (bins - 1)
            # cv2.threshold(hist, min_val, max_val, cv2.THRESH_BINARY, cv2.CV_8UC1)
            # cv2.imwrite(os.path.join(output_path, "cv_" + file), cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (256, 256)))


if __name__ == '__main__':
    imageEnhance(input_path, output_path)
