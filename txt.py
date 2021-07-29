import cv2
import math
import numpy as np
import time

time_start = time.time()
input_img_file = r"C:\Users\lenovo\Desktop\IMG_20210620_104517.jpg"

# Picture reading processing
def ReadPic(input_img_file):
    image = cv2.imread(input_img_file)
    scala = int(1000 / image.shape[1])
    if scala == 0:
        scala = int(image.shape[1] / 1000)
        image = cv2.resize(image, (int(image.shape[1] / scala), int(image.shape[0] / scala)))
    else:
        image = cv2.resize(image, (int(image.shape[1] * scala), int(image.shape[0] * scala)))
    return image

# Conversion degree
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res

# Rotate the image degree angle counterclockwise (full size)
def RotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    #print(RotateMatrix)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate

# The angle is calculated by Hough transform
def CalcDegree(srcImage):
    midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(midImage, 50, 200, 3)
    lineimage = srcImage.copy()

    # Line detection by Hough transform
    # The fourth parameter is the threshold. The larger the threshold, the higher the detection accuracy
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 150)
    # Due to different images, the threshold is not easy to set, because the threshold setting is too high, so it is impossible to detect straight lines.
    # The threshold is too low, there are too many straight lines, and the speed is very slow
    sum = 0
    # Draw each line segment in turn
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # Only the smallest angle is selected as the rotation angle
            sum += theta
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.imshow("Imagelines", lineimage)

    # Average all angles so that the rotation effect is better
    average = sum / len(lines)
    angle = DegreeTrans(average) - 90
    return angle

#Location one-dimensional code
def LocateCode(gray):
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    level = 225
    while level > 0:
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, level, 255, cv2.THRESH_BINARY)
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        except:
            level -= 25
            continue
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        i = 1
        xul = []
        while i < 4:
            lenx = box[i][0] - box[0][0]
            leny = box[i][1] - box[0][1]
            lens = math.sqrt(lenx * lenx + leny * leny)
            xul.append(lens)
            i += 1
        xul.sort()
        a = xul[1] / xul[0]
        if (a > 3.4 and a < 3.5) or (a > 4.6 and a < 5.0) or (a > 5.1 and a < 7.75):
            break
        level -= 25
    newbox0 = sorted(box, key=lambda x: x[0])
    newbox1 = sorted(box, key=lambda x: x[1])
    return newbox0, newbox1

if __name__ == '__main__':
    image = ReadPic(input_img_file)
    try:
        degree = CalcDegree(image)
        rotate = RotateImage(image, degree)
    except:
        rotate = image
    gray = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
    box0, box1 = LocateCode(gray)
    dst = gray[box1[0][1]:box1[3][1], box0[0][0]:box0[3][0]]

    cv2.imwrite(r"C:\Users\lenovo\Desktop\Answer.jpg", dst)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
