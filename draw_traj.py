from Generator import Generator_net
from readFIle import loader_train, endpoint_calssify, image2world, H,  dset_train
from GCN import GraphConv, Graph_normalize
from transformer_xl import MemTransformerLM
import torch
from time_expand_conv import TxpCnn
from tcn import TemporalConvNet
import torchvision.models as models
from torch import optim, nn
from readImg import img, centres, EndPoint_Pred_CNN
from contex_cnn import ContexConv
from Discriminator import Discriminator_net
import torch.nn.functional as F
import time
import random
from File_name import filename, reference_path
import cv2
import numpy as np
GCN_inp_d = 2
GCN_hid1_d = 10
GCN_hid2_d = 50
GCN_out_d = 60

contex_cnn_out_size = [1, 40]

loss_batch = 0

mem_len = 4
n_token = 10000
n_layer = 2
n_head = 2
d_model = GCN_out_d + contex_cnn_out_size[1]
d_head = d_model
d_inner = 300
dropout = 0
ext_len = 0

txp_inseq = 8
txp_outseq = 12

classfy_cnn_input_size = [8, 1, 12, 100]  # 只用到第二个维度的长度
num_class = 30

tcn_input_channel = 102
# tcn_channels = [102, 300, 200, 100, 32, 2]
tcn_channels = [tcn_input_channel, 64, 16, 2]

GRU_input_size = 2
GRU_hidden_size = contex_cnn_out_size[1]
GRU_num_layers = 1
'''
param_epoch_1 = 49
param_epoch_2 = 49
param_epoch_3 = 49
param_epoch_4 = 199
param_epoch_5 = 199
param_epoch_6 = 199
cross_valid_num = 6
'''
k_temp = 4
param_epoch = 199
k = 6



img_path = reference_path

img_scene = cv2.imread(img_path)

imgZero = np.zeros(img_scene.shape, np.uint8)
imgZero[:] = 255

if filename == 'zara01' or filename == 'zara02' or filename == 'zara03' or filename == 'students01' or \
    filename == 'students03':
    loc_mode = 1
elif filename == 'eth' or filename == 'htl':
    loc_mode = 2
else:
    raise Exception("don't have that dataset")


def valid(k, k_temp, centres, model_G,  img):
    batch_count = 0

    '''
    对于数据（索引）的处理:分出验证集与测试集，打乱测试集顺序
    '''
    # 计算用于训练及测试的样本index
    sample_per_part = (dset_train.__len__() // k) + 1
    # print(sample_per_part)
    # 生成完整索引
    index_full = [i for i in range(dset_train.__len__())]
    # 分为train part与test part

    index_test = index_full[sample_per_part * (k_temp - 1):
                            sample_per_part * k_temp]

    index_test = index_test[:100]

    img_scene = cv2.imread(img_path)
    imgZero = np.zeros(img_scene.shape, np.uint8)
    imgZero[:] = 255

    '''
    ********************数据（索引）处理完毕******************
    '''
    for index in index_test:
        '''
        
        '''

        batch = dset_train.__getitem__(index)
        batch_count += 1

        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        obs_traj.unsqueeze_(0)
        pred_traj_gt.unsqueeze_(0)
        V_obs.unsqueeze_(0)
        A_obs.unsqueeze_(0)
        V_tr.unsqueeze_(0)
        A_tr.unsqueeze_(0)

        centres = torch.as_tensor(centres).cuda()
        endpoint_class_orig = endpoint_calssify(pred_traj_gt, centres)
        endpoint_class = endpoint_class_orig.squeeze_().int()  # 用于onehot编码
        endpoint_class_forG = endpoint_class_orig.type(torch.LongTensor).cuda()
        pred_traj_gt.squeeze_()

        V_obs = V_obs.squeeze()
        A_obs = A_obs.squeeze()
        V_tr.squeeze_()

        classfy_pred, traj_pred, context = model_G(V_obs, A_obs, img, centres)

        classfy_pred.detach_()
        traj_pred.detach_()
        context.detach_()

        obs_traj = obs_traj.squeeze().permute(0, 2, 1)
        pred_traj_gt = pred_traj_gt.squeeze().permute(0, 2, 1)
        traj_pred = traj_pred.permute(0, 2, 1)

        print(pred_traj_gt.shape)
        # torch.Size([n, 12, 2])
        print(obs_traj.shape)
        # torch.Size([n, 8, 2])
        print(traj_pred.shape)
        # torch.Size([n, 12, 2])
        for i, trajs in enumerate(zip(obs_traj, pred_traj_gt, traj_pred)):
            obs, pred, traj_pred = trajs
            if i >= 5:
                continue
            x1, y1 = -1, -1
            for loc in obs:

                if x1 == -1 or y1 == -1:
                    if loc_mode == 1:
                        x1, y1 = loc
                    elif loc_mode == 2:
                        y1, x1 = loc
                    y1 = int(img_scene.shape[0] - y1)
                    continue
                if loc_mode == 1:
                    x2, y2 = loc
                elif loc_mode == 2:
                    y2, x2 = loc
                y2 = int(img_scene.shape[0] - y2)
                cv2.line(img_scene, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
                cv2.line(imgZero, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)

                cv2.circle(img_scene, (x2, y2), radius=2, color=(255, 0, 0), thickness=-1)  # , shift=1)
                cv2.circle(imgZero, (x2, y2), radius=2, color=(255, 0, 0), thickness=-1)  # , shift=1)

                x1, y1 = x2, y2
            obs_end_x = x1
            obs_end_y = y1
            for loc in pred:
                if loc_mode == 1:
                    x2, y2 = loc
                elif loc_mode == 2:
                    y2, x2 = loc
                y2 = int(img_scene.shape[0] - y2)
                cv2.line(img_scene, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
                cv2.line(imgZero, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)

                cv2.circle(img_scene, (x2, y2), radius=2, color=(255, 0, 0), thickness=-1)  # , shift=1)
                cv2.circle(imgZero, (x2, y2), radius=2, color=(255, 0, 0), thickness=-1)  # , shift=1)

                x1, y1 = x2, y2
            x1 = obs_end_x
            y1 = obs_end_y
            for loc in traj_pred:
                if loc_mode == 1:
                    x2, y2 = loc
                elif loc_mode == 2:
                    y2, x2 = loc
                y2 = int(img_scene.shape[0] - y2)
                cv2.line(img_scene, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
                cv2.line(imgZero, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)

                cv2.circle(img_scene, (x2, y2), radius=2, color=(0, 0, 255), thickness=-1)#, shift=1)
                cv2.circle(imgZero, (x2, y2), radius=2, color=(0, 0, 255), thickness=-1)#, shift=1)

                x1, y1 = x2, y2
            '''
            cv2.imshow("image-lines{:d}".format(index), img_scene)
            cv2.imshow("image-zero-lines{:d}".format(index), imgZero)
            cv2.waitKey(10)
            '''
            # cv2.imwrite("traj_pic{:d}.jpg".format(index), img_scene)
            # cv2.imwrite("traj_zero_pic{:d}.jpg".format(index), imgZero)
            # return
        cv2.imwrite("traj_pic{:d}.jpg".format(index), img_scene)
        cv2.imwrite("traj_zero_pic{:d}.jpg".format(index), imgZero)
        img_scene = cv2.imread(img_path)
        imgZero = np.zeros(img_scene.shape, np.uint8)
        imgZero[:] = 255

    return


if __name__ == "__main__":
    loss_batch = 0
    batch_count = 0
    ade = 0.0
    fde = 0.0
    torch.cuda.set_device(0)
    if not torch.cuda.is_available():
        raise Exception("GPU is not available")
    model_G = Generator_net(GCN_inp_d, GCN_hid1_d, GCN_hid2_d, GCN_out_d,
                            n_token, n_layer, n_head, d_model, d_head,
                            d_inner, dropout, mem_len, ext_len,
                            txp_inseq, txp_outseq,
                            contex_cnn_out_size,
                            classfy_cnn_input_size, num_class,
                            tcn_input_channel, tcn_channels).cuda()
    model_D = Discriminator_net(GRU_input_size, GRU_hidden_size, GRU_num_layers).cuda()

    model_G_param = torch.load('./savepoint/{}/generator_temp_cross{:d}_{:d}.pth'.
                                format(filename, k_temp, param_epoch), map_location='cuda:0')
    model_D_param = torch.load('./savepoint/{}/discriminator_temp_cross{:d}_{:d}.pth'
                                .format(filename, k_temp, param_epoch), map_location='cuda:0')
    model_G.load_state_dict(model_G_param)
    model_D.load_state_dict(model_D_param)
    valid(k=6, k_temp=k_temp, centres=centres, model_G=model_G, img=img)
