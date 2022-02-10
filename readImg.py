import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torchvision.models as models
from torch import nn
import math
from File_name import img_path, reference_path
from tcn import TemporalConvNet

# 测试直接输入原始图像
img_path = reference_path
# cv2.imread()------np.array, (H x W xC), [0, 255], BGR
img_cv = cv2.imread(img_path)
# print(img_cv.shape)
# print(htl_cv.shape)

# ------------np.ndarray转为torch.Tensor------------------------------------
# numpy image: H x W x C
# torch image: C x H x W
# np.transpose( xxx,  (2, 0, 1))   # 将 H x W x C 转化为 C x H x W
img = torch.from_numpy(np.transpose(img_cv, (2, 0, 1))).to(torch.float32)  # [3, 480, 640]

img.unsqueeze_(0)
print("img size", img.shape)


def image_split(img):
    # 输入一张图像，输入分割后的各中心点坐标矩阵
    height_num = 5
    width_num = 6
    height = img.shape[-2]
    width = img.shape[-1]
    # input shape: [batch, channel, height, weight]
    # 分成 宽10、长10，共100份
    per_height = height / height_num
    per_width = width / width_num

    # print(per_height, per_width)
    centres = np.zeros([height_num, width_num, 2])
    for i in range(height_num):
        for j in range(width_num):
            x = per_width * ((2*j + 1) / 2)
            y = per_height * ((2*i + 1) / 2)
            centres[i, j] = [y, x]
    # print(centres)
    return centres


def Modify_Res50(class_num):
    resnet50 = models.resnet18(pretrained=False)
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, class_num),
        nn.LogSoftmax(dim=1)
    )
    return resnet50


class EndPoint_Pred_CNN(nn.Module):

    def __init__(self, input_shape, num_class):
        super(EndPoint_Pred_CNN, self).__init__()
        self.num_class = num_class
        self.input_shape = input_shape
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(input_shape[1], 3, (1, 20)))
        self.layers.append(Modify_Res50(num_class))

    def forward(self, input):
        out1 = self.layers[0](input)
        out2 = self.layers[1](out1)
        return out2

'''
class EndPoint_Pred_TCN(nn.Module):
    def __init__(self, tcn_input_channel, tcn_channels):
        super(EndPoint_Pred_TCN, self).__init__()
        self.tcn_net = TemporalConvNet(tcn_input_channel, tcn_channels)
        self.conv = nn.Conv1d()
'''

centres = image_split(img)
# print("Centres", centres.shape)


if __name__ == '__main__':
    img = torch.rand(3, 480, 640)
    a = image_split(img)
    # model = EndPoint_Pred_CNN(img.shape, 30)
    # res = model(img)
    # print(res.shape)
    # [5, 30]









