import cv2

if __name__ == '__main__':
    image_path = "/data/ldp/zjf/infer/00270.jpg"  # 输入的RGB图像文件路径
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite('/data/ldp/zjf/infer/output.jpg', result)


