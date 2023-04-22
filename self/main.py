import cv2, os
import numpy as np
from PIL import Image, ImageEnhance, ImageChops


# 灰度图转换
def grayscale(image):
    result = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return result


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
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # cv2.imwrite("/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/res/img_mask.jpg", mask)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


if __name__ == "__main__":
    output_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/res"
    img_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/img/1492626274615008344/1.jpg"
    # img_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/img/00060.jpg"
    gray_path = os.path.join(output_path, "img_gray.jpg")
    gauss_path = os.path.join(output_path, "img_gauss.jpg")
    canny_path = os.path.join(output_path, "img_canny.jpg")
    mask_path = os.path.join(output_path, "img_mask.jpg")
    contrasted_path = os.path.join(output_path, "img_contrasted.jpg")
    subtract_path = os.path.join(output_path, "img_subtract.jpg")
    bright_path = os.path.join(output_path, "img_bright.jpg")
    bright_path2 = os.path.join(output_path, "img_bright2.jpg")
    canny_path2 = os.path.join(output_path, "img_canny2.jpg")
    brightCanny_path = os.path.join(output_path, "img_brightCanny.jpg")
    gray_path2 = os.path.join(output_path, "img_gray2.jpg")
    brightGray_path = os.path.join(output_path, "img_brightGray.jpg")
    line_path = os.path.join(output_path, "img_line.jpg")

    img = cv2.imread(img_path)
    # 转灰度图像
    gray = grayscale(img)
    cv2.imwrite(gray_path, gray)
    # 高斯滤波
    gauss = cv2.GaussianBlur(gray, (5, 5), 0, 0)
    cv2.imwrite(gauss_path, gauss)
    # Canny边缘检测
    canny = cv2.Canny(gauss, 75, 225)
    cv2.imwrite(canny_path, canny)
    # mask掩膜
    p1 = [0, 240]
    p2 = [0, 719]
    p3 = [1279, 719]
    p4 = [1279, 240]
    vertices = np.array([p1, p2, p3, p4], dtype=np.int32)
    mask = region_of_interest(canny, [vertices])
    cv2.imwrite(mask_path, mask)

    img_gray = Image.open(gray_path, 'r')
    img_contrasted = ImageEnhance.Contrast(img_gray).enhance(2)
    img_contrasted.save(contrasted_path)

    img_contrasted = cv2.imread(contrasted_path)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 60, np.array([]), minLineLength=30,
                            maxLineGap=40)
    # test1较优参数
    for num in range(len(lines)):
        for x1, y1, x2, y2 in lines[num]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(line_path, img)

    # lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, 100, np.array([]), minLineLength=100,
    #                         maxLineGap=8)  # ta参



    # bright = cv2.imread(bright_path)
    # Thresh = 240
    # # 对图片进行阈值处理
    # bright = cv2.threshold(bright, Thresh, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite(bright_path2, bright)
    # gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite(gray_path2, gray)

    # bright_gray = grayscale(bright)
    #
    # subtract = ImageChops.subtract(bright_gray, img_gray, scale=3, offset=0)
    # subtract.save(subtract_path)

    # canny[np.where(bright_gray == 0)] = 0
    #
    # brightCanny = cv2.Canny(grayscale(bright), 75, 225)
    # cv2.imwrite(brightCanny_path, brightCanny)
    #
    # pos = np.where(brightCanny == 255)
    # canny[pos] = 0
    # cv2.imwrite(canny_path2, canny)


    # gray[pos] = 128
    # pos2 = np.where(gray == 0)
    # # print(pos == pos2)
    # cv2.imwrite(canny_path2, gray)
    # p1 = cv2.imread(gray_path)
    # p2 = cv2.imread(contrasted_path)
    # subtract = cv2.subtract(p1, p2)
    # # cv2.imwrite(subtract_path, subtract)
    # p3 = cv2.Canny(p2, 75, 225)
    # cv2.imwrite(subtract_path, p3)

    # subtract = ImageChops.subtract(contrasted, img_2, scale=3, offset=0)
    # subtract.save(subtract_path)

    #
    # img_bright = Image.open(img_path, 'r')
    # img_bright = ImageEnhance.Brightness(img_bright).enhance(3)
    # img_bright.save(bright_path)
    #
    # bright = cv2.imread(bright_path)
    # bright_gray = grayscale(bright)
    # cv2.imwrite(brightGray_path, bright_gray)
    # pos = np.where(bright_gray < 250)
    # bright_gray[pos] = gray[pos]
    # cv2.imwrite(bright_path2, bright_gray)
    #
    # bright_gray = Image.open(brightGray_path, 'r')
    # img_gray = Image.open(gray_path, 'r')
    # subtract = ImageChops.subtract(bright_gray, img_gray, scale=1, offset=0)
    # subtract.save(subtract_path)
