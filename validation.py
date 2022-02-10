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

param_epoch_1 = 49
param_epoch_2 = 49
param_epoch_3 = 49
param_epoch_4 = 199
param_epoch_5 = 199
param_epoch_6 = 199
cross_valid_num = 6


def ADE(pred, target):
    loss = nn.L1Loss(reduction='mean')
    return loss(pred, target)


def FDE(pred, target):
    pred = pred[:, :, -1]
    target = target[:, :, -1]
    loss = nn.L1Loss(reduction='mean')
    return loss(pred, target)


def criterion_G_classfy(classfy_pred, endpoint_class):
    # (endpoint_class.device)
    loss = F.nll_loss(classfy_pred, endpoint_class).cuda()
    return loss


def criterion_G_traj(traj_pred, pred_traj_gt, D_output, D_real_label):
    loss_gan = 10 * F.binary_cross_entropy(D_output, D_real_label).cuda()
    loss_wta = 0.0001 * F.mse_loss(traj_pred, pred_traj_gt).cuda()
    return loss_gan + loss_wta, loss_wta


def valid(k, k_temp, centres, num_class, model_G, model_D, img, GRU_num_layers,
          contex_cnn_out_size):
    batch_count = 0
    total_loss_d = 0
    total_loss_g = 0
    total_ADE = 0
    total_FDE = 0
    criterion_D = nn.BCELoss().cuda()

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

    '''
    ********************数据（索引）处理完毕******************
    '''
    for index in index_test:
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

        D_real_label = torch.ones(V_tr.shape[1], 1).cuda()  # 定义真的label为1
        D_fake_label = torch.zeros(V_tr.shape[1], 1).cuda()  # 定义假的label为0

        classfy_pred, traj_pred, context = model_G(V_obs, A_obs, img, centres)
        # print('pred_traj_gt', pred_traj_gt)
        # print('traj_pred', traj_pred)
        classfy_pred.detach_()
        traj_pred.detach_()
        context.detach_()
        # ***将轨迹从像素坐标系转换到世界坐标系*************
        traj_pred_world = traj_pred.cpu()
        pred_traj_gt_world = pred_traj_gt.cpu()
        traj_pred_arr = traj_pred_world.numpy()
        pred_traj_gt_arr = pred_traj_gt_world.numpy()
        for i, arr in enumerate(traj_pred):
            traj_pred_arr[i] = image2world(traj_pred_arr[i].T, H).T
            pred_traj_gt_arr[i] = image2world(pred_traj_gt_arr[i].T, H).T
        traj_pred_world = torch.from_numpy(traj_pred_arr).cuda()
        pred_traj_gt_world = torch.from_numpy(pred_traj_gt_arr).cuda()
        traj_pred = traj_pred_world
        pred_traj_gt = pred_traj_gt_world
        # print('traj_pred_world', traj_pred_world)
        # print('pred_traj_gt_world', pred_traj_gt_world)
        # ************转换结束****************************

        context_expand = torch.zeros(GRU_num_layers, V_tr.shape[1],
                                     contex_cnn_out_size[1]).cuda()
        context_expand[:, :] = context

        real_out = model_D(V_tr, context_expand)
        real_out.detach_()
        # [n, 1]
        d_loss_real = criterion_D(real_out, D_real_label)  # 得到真实轨迹的loss
        real_scores = real_out  # 得到真实轨迹的判别值，输出的值越接近1越好

        # 计算假的轨迹的损失
        fake_out = model_D(traj_pred.permute(2, 0, 1), context_expand)
        fake_out.detach_()
        d_loss_fake = criterion_D(fake_out, D_fake_label)  # 得到假的轨迹的loss
        fake_scores = fake_out  # 得到假轨迹的判别值，对于判别器来说，假轨迹的损失越接近0越好

        d_loss = d_loss_real + d_loss_fake

        g_loss_classfy = criterion_G_classfy(classfy_pred, endpoint_class_forG)
        # g_loss_traj = criterion_G_traj(traj_pred, pred_traj_gt, fake_out, D_real_label)

        g_loss_wta = F.mse_loss(traj_pred, pred_traj_gt).cuda()
        g_loss = g_loss_classfy + g_loss_wta
        loss_ade = ADE(traj_pred, pred_traj_gt)
        loss_fde = FDE(traj_pred, pred_traj_gt)
        '''

        print('d_real{:.6f}, d_fake{:.6f}, g_classfy:{:.6f}, g_traj:{:.6f} '
              'ADE{:.6f}, FDE{:.6f}'.format(d_loss_real, d_loss_fake, g_loss_classfy,
                                            g_loss_wta, loss_ade, loss_fde))
        '''
        total_loss_d = d_loss + total_loss_d
        total_loss_g = g_loss + total_loss_g
        total_ADE = total_ADE + loss_ade
        total_FDE = total_FDE + loss_fde

    total_loss_g = total_loss_g / batch_count
    total_loss_d = total_loss_d / batch_count
    total_FDE = total_FDE / batch_count
    total_ADE = total_ADE / batch_count

    print('total_loss_d:{:.6f}, total_loss_g:{:.6f} '
          'total_ADE:{:.6f}, total_FDE{:.6f}'.format(total_loss_d, total_loss_g,
                                                     total_ADE, total_FDE))

    return total_ADE, total_FDE


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
    for k_temp in range(1, cross_valid_num + 1):
        if k_temp == 1:
            param_epoch = param_epoch_1
        elif k_temp == 2:
            param_epoch = param_epoch_2
        elif k_temp == 3:
            param_epoch = param_epoch_3
        elif k_temp == 4:
            param_epoch = param_epoch_4
        elif k_temp == 5:
            param_epoch = param_epoch_5
        elif k_temp == 6:
            param_epoch = param_epoch_6
        else:
            raise RuntimeError("wrong k_temp:{:d}".format(k_temp))
        model_G_param = torch.load('./savepoint/generator_temp_cross{:d}_{:d}.pth'.
                                   format(k_temp, param_epoch), map_location='cuda:0')
        model_D_param = torch.load('./savepoint/discriminator_temp_cross{:d}_{:d}.pth'
                                   .format(k_temp, param_epoch), map_location='cuda:0')
        model_G.load_state_dict(model_G_param)
        model_D.load_state_dict(model_D_param)
        a, b = valid(k=cross_valid_num, k_temp=k_temp, centres=centres, num_class=num_class, model_G=model_G,
                     model_D=model_D, img=img, GRU_num_layers=GRU_num_layers,
                     contex_cnn_out_size=contex_cnn_out_size)
        ade += a
        fde += b
    fde /= cross_valid_num
    ade /= cross_valid_num
    print("total fed loss is: ", fde)
    print("total ade loss is: ", ade)
