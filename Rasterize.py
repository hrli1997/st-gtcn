import cv2
from File_name import reference_path
import numpy as np


def whiteful(img):
    rows, cols = img.shape[:2]
    SIZE = 3  # 卷积核大小
    P = int(SIZE / 2)
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    BEGIN = False
    BP = []

    for row in range(P, rows - P, 1):
        for col in range(P, cols - P, 1):
            # print(img[row,col])
            if (img[row, col] == WHITE).all():
                kernal = []
                for i in range(row - P, row + P + 1, 1):
                    for j in range(col - P, col + P + 1, 1):
                        kernal.append(img[i, j])
                        if (img[i, j] == BLACK).all():
                            # print(i,j)
                            BP.append([i, j])

    print('BP', len(BP))
    uniqueBP = np.array(list(set([tuple(c) for c in BP])))
    print('uniqueBP', len(uniqueBP))

    for x, y in uniqueBP:
        img[x, y] = WHITE
    return img


# robert 算子[[-1,-1],[1,1]]
def robert_canny(img):
    r, c = img.shape
    r_sunnzi = [[-1, -1], [1, 1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x + 2, y:y + 2]
                list_robert = r_sunnzi * imgChild
                img[x, y] = abs(list_robert.sum())  # 求和加绝对值
    return img


# sobel算子的实现
def sobel_canny(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                                       new_imageY[i + 1, j + 1]) ** 0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image)  # 无方向算子处理的图像


# Laplace算子
# 常用的Laplace算子模板 [[0,1,0],[1,-4,1],[0,1,0]]  [[1,1,1],[1,-8,1],[1,1,1]]
def Laplace_canny(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_image[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * L_sunnzi))
    return np.uint8(new_image)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv2.HoughLinesP(image, 1, np.pi / 90, 60, minLineLength=100, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv2.imshow("line_detect_possible_demo", image)
    cv2.waitKey(8000)
    cv2.destroyAllWindows()

#标准霍夫线变换
def line_detection(image):
 # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
 # edges = cv2.Canny(gray, 50, 150, apertureSize=3) #apertureSize参数默认其实就是3
 # cv2.imshow("edges", image)
 lines = cv2.HoughLines(image, 1, np.pi/180, 10)
 for line in lines:
    rho, theta = line[0] #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
    a = np.cos(theta) #theta是弧度
    b = np.sin(theta)
    x0 = a * rho #代表x = r * cos（theta）
    y0 = b * rho #代表y = r * sin（theta）
    x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
    y1 = int(y0 + 1000 * a) #计算起始起点纵坐标
    x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
    y2 = int(y0 - 1000 * a) #计算直线终点纵坐标 注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10) #点的坐标必须是元组，不能是列表。
 cv2.imshow("image-lines", image)
 cv2.waitKey(8000)
 cv2.destroyAllWindows()

IMAGE_PATH = reference_path
IMAGE_NAME = IMAGE_PATH.split(sep='/')[-1]
SAVE_IMAGE_NAME = "canny_" + IMAGE_NAME
whiteful_times = 5
# ****************边缘提取
img = cv2.imread(IMAGE_PATH)
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
c_canny_img = cv2.Canny(img2gray, 50, 150)
'''
cv2.imshow('mask', c_canny_img)
k = cv2.waitKey(10) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
'''
cv2.imwrite(SAVE_IMAGE_NAME, c_canny_img)

# line_detect_possible_demo(c_canny_img)
exit()
# ***************填白

img = cv2.imread(SAVE_IMAGE_NAME)
for i in range(whiteful_times):
    img = whiteful(img)
cv2.imwrite('second_bird.png', img)
cv2.imshow('img', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
