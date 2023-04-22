import cv2
import numpy as np
from PIL import Image

if __name__ == "__main__":
    img_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/img/600.jpg"
    res_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/segment/res.jpg"
    crop_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/segment/crop.jpg"
    mask_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/self/segment/mask.jpg"

    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    im = img.load()
    for i in range(img.width):
        for j in range(int(img.height / 3)):
            im[i, j] = (0, 0, 0)
    img.save(res_path)


    # img[np.where(img != 0)] = 255
    # cv2.imwrite(res_path, img)

    # CULane
    # p1 = [0, 0]
    # p2 = [0, 200]
    # p3 = [1640, 200]
    # p4 = [1640, 0]
    # vertices = np.array([p1, p2, p3, p4], dtype=np.int32)    # 将区域填充为黑色
    # cv2.fillPoly(img, [vertices], (0, 0, 0))
    # cv2.imwrite(mask_path, img)


