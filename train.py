from Generator import Generator_net, single_output_train
from readFIle import loader_train, endpoint_calssify, loader_val, loader_test
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
import hiddenlayer as hl
import os

GCN_inp_d = 2
GCN_hid1_d = 10
GCN_hid2_d = 50
GCN_hid3_d = 300
GCN_hid4_d = 300
GCN_out_d = 60

contex_cnn_out_size = [1, 40]

loss_batch = 0
batch_count = 0

mem_len = 4
n_token = 10000
n_layer = 5
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
tcn_channels = [102, 300, 300, 200, 100, 32, 2]

GRU_input_size = 2
GRU_hidden_size = contex_cnn_out_size[1]
GRU_num_layers = 1

num_epoch = 500
batch_size = 10

# 是否加载预训练模型
load_pretrained = False
D_param = 18  # last valid epoch:27
G_param = D_param
single_output_train = single_output_train # 若为True，则为单输出训练，梯度仅从单个输出（轨迹输出）反向传播
# 若为False，则为双输出，梯度从两个输出同时传播


def criterion_G_classfy(classfy_pred, endpoint_class, reduction='mean'):
    pos = torch.eq(endpoint_class, 1).float()
    neg = torch.eq(endpoint_class, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    return 100 * F.binary_cross_entropy_with_logits(classfy_pred, endpoint_class, weights, reduction=reduction)


def criterion_G_traj(traj_pred, pred_traj_gt, D_output, D_real_label):
    loss_gan = 1 * F.binary_cross_entropy(D_output, D_real_label).cuda()
    loss_wta = 1 * F.mse_loss(traj_pred, pred_traj_gt).cuda()
    return loss_gan + loss_wta, loss_wta


def one_hot_encode(lables, num_class):
    """
    :param lables: 一维数组，如[1, 4, 5, 7]
    :param num_class: int型数据
    :return: 二维数组，[len(lable), num_class]
    """
    one_hot = torch.zeros(len(lables), num_class).cuda()
    for i, lable in enumerate(lables):
        one_hot[i, lable] = 1
    return one_hot


def valid(centres, num_class, model_G, model_D, img, GRU_num_layers,
          contex_cnn_out_size, history_val, canvas_val, epoch):

    batch_count = 0
    total_loss_d = 0
    total_loss_g_wta = 0
    total_loss_g_class = 0
    total_ADE = 0
    total_FDE = 0
    criterion_D = nn.BCELoss().cuda()
    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch


        centres = torch.as_tensor(centres).cuda()
        endpoint_class_orig = endpoint_calssify(pred_traj_gt, centres)
        endpoint_class = endpoint_class_orig.squeeze_().int()  # 用于onehot编码
        endpoint_class_forG = endpoint_class_orig.type(torch.LongTensor).cuda()
        gt_class_one_hot = one_hot_encode(endpoint_class, num_class)
        pred_traj_gt.squeeze_()

        V_obs = V_obs.squeeze()
        A_obs = A_obs.squeeze()
        V_tr.squeeze_()

        D_real_label = torch.ones(V_tr.shape[1], 1).cuda()  # 定义真的label为1
        D_fake_label = torch.zeros(V_tr.shape[1], 1).cuda()  # 定义假的label为0

        classfy_pred, traj_pred, context = model_G(V_obs, A_obs, img, centres)
        classfy_pred.detach_()
        traj_pred.detach_()
        context.detach_()
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

        g_loss_classfy = criterion_G_classfy(classfy_pred, gt_class_one_hot)
        # g_loss_traj = criterion_G_traj(traj_pred, pred_traj_gt, fake_out, D_real_label)
        g_loss_wta = F.mse_loss(traj_pred, pred_traj_gt).cuda()
        g_loss = g_loss_classfy + g_loss_wta
        # loss_ade = ADE(traj_pred, pred_traj_gt)
        # loss_fde = FDE(traj_pred, pred_traj_gt)

        # print('d_real{:.6f}, d_fake{:.6f}, g_classfy:{:.6f}, g_traj:{:.6f} '
        #      .format(d_loss_real, d_loss_fake, g_loss_classfy, g_loss_wta))

        total_loss_d = d_loss + total_loss_d
        total_loss_g_wta = g_loss_wta + total_loss_g_wta
        total_loss_g_class = g_loss_classfy + total_loss_g_class

    total_loss_g_wta = total_loss_g_wta / batch_count
    total_loss_g_class = total_loss_g_class / batch_count
    total_loss_d = total_loss_d / batch_count
    print("Validation:")
    print('total_loss_d:{:.6f}, total_loss_g_wta:{:.6f} '
          'total_loss_class{:.6f}'.format(total_loss_d, total_loss_g_wta, total_loss_g_class))

    history_val.log((epoch),
                D_loss=total_loss_d,
                g_traj_loss=total_loss_g_wta,
                g_class_loss=total_loss_g_class)
    # 可视化
    with canvas_val:
        canvas_val.draw_plot(history_val["D_loss"])
        canvas_val.draw_plot(history_val["g_traj_loss"])
        canvas_val.draw_plot(history_val["g_class_loss"])
    canvas_val.save(os.path.join('./savepoint/', "validation_progress.png"))
    return



def train(batch_count, centres, num_class, model_G, model_D, img, GRU_num_layers,
          contex_cnn_out_size):
    history = hl.History()
    canvas = hl.Canvas()

    history_epoch = hl.History()
    canvas_epoch = hl.Canvas()

    criterion_D = nn.BCELoss().cuda()  # 单目标二分类交叉熵函数
    d_optimizer = torch.optim.Adam(model_D.parameters(), lr=0.00001)
    # d_optimizer = torch.optim.SGD(model_D.parameters(), lr=0.00001, momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False)
    g_optimizer = torch.optim.Adam(model_G.parameters(), lr=0.0000005)
    # g_optimizer = torch.optim.SGD(model_G.parameters(), lr=0.000000001, )
                                 # momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False)
    # g_optimizer = torch.optim.RMSprop(model_G.parameters(), lr=0.0001, alpha=0.6)
    global num_epoch
    model_G.train()
    model_D.train()
    for epoch in range(num_epoch):  # 进行多个epoch的训练
        total_G_loss_wta = 0
        total_G_loss_class = 0
        cnts = 0
        for cnt, batch in enumerate(loader_test):
            time_start = time.time()
            batch_count += 1

            # Get data
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            centres = torch.as_tensor(centres).cuda()
            endpoint_class_orig = endpoint_calssify(pred_traj_gt, centres)
            endpoint_class = endpoint_class_orig.squeeze_().int()  # 用于onehot编码
            endpoint_class_forG = endpoint_class_orig.type(torch.LongTensor).cuda()

            # 用于生成器的损失函数
            # [n]
            pred_traj_gt.squeeze_()
            # 将真实终点分类化为one-hot类型，用于计算分类网络损失
            gt_class_one_hot = one_hot_encode(endpoint_class, num_class)
            # [n, 100]
            # 对矩阵形状进行处理
            V_obs = V_obs.squeeze()
            A_obs = A_obs.squeeze()
            V_tr.squeeze_()
            # print("V_tr", V_tr.shape, V_tr)
            # [12, n, 2]
            D_real_label = torch.ones(V_tr.shape[1], 1).cuda()  # 定义真的label为1
            D_fake_label = torch.zeros(V_tr.shape[1], 1).cuda()  # 定义假的label为0
            # print('D_fake_label1', D_fake_label)
            # print('D_real_label1', D_real_label)
            # [n, 1]

            # ########判别器训练train#####################
            # 分为两部分：1、真的轨迹判别为真；2、假的轨迹判别为假
            classfy_pred, traj_pred, context = model_G(V_obs, A_obs, img, centres)
            traj_pred = traj_pred.detach()
            # [n, 2, 12]
            traj_pred = traj_pred.permute(2, 0, 1)
            context = context.detach()
            context_expand = torch.zeros(GRU_num_layers, V_tr.shape[1],
                                         contex_cnn_out_size[1]).cuda()
            context_expand[:, :] = context
            # [1, n, 40]

            # 计算真实轨迹的损失
            real_out = model_D(V_tr, context_expand)
            # [n, 1]
            d_loss_real = criterion_D(real_out, D_real_label)  # 得到真实轨迹的loss
            real_scores = real_out  # 得到真实轨迹的判别值，输出的值越接近1越好

            # 计算假的轨迹的损失
            fake_out = model_D(traj_pred, context_expand)
            d_loss_fake = criterion_D(fake_out, D_fake_label)  # 得到假的轨迹的loss
            fake_scores = fake_out  # 得到假轨迹的判别值，对于判别器来说，假轨迹的损失越接近0越好

            # 损失函数和优化
            d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
            d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            d_loss.backward()  # 将误差反向传播
            d_optimizer.step()  # 更新参数

            # ###############################生成网络的训练###############################
            # 原理：目的是希望生成的假的轨迹被判别器判断为真的图片，
            # 在此过程中，将判别器固定，将假的轨迹传入判别器的结果与真实的label对应，
            # 反向传播更新的参数是生成网络里面的参数，
            # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的轨迹让判别器以为是真的
            # 这样就达到了对抗的目的
            # 计算假的轨迹的损失
            classfy_pred, traj_pred, _ = model_G(V_obs, A_obs, img, centres)
            D_output = model_D(traj_pred.permute(2, 0, 1), context_expand)
            D_output.detach_()
            g_loss_classfy = criterion_G_classfy(classfy_pred, gt_class_one_hot)
            g_loss_traj, g_loss_wta = criterion_G_traj(traj_pred, pred_traj_gt, D_output, D_real_label)

            if not single_output_train:
                g_loss = g_loss_traj + g_loss_classfy
            else:
                g_loss = g_loss_traj
            # bp and optimize
            g_optimizer.zero_grad()  # 梯度归0
            torch.autograd.set_detect_anomaly(True)
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
            total_G_loss_wta += g_loss_wta
            total_G_loss_class += g_loss_classfy

            cnts = cnt
            if (cnt + 1) % 5 == 0:
                time_end = time.time()
                print('Epoch[{}/{}], used_time:{:.3f}'.format(
                    epoch, num_epoch, time_end - time_start  # 打印的是真实轨迹的损失均值
                ))
                print('d_loss:{:.6f}, g_loss:{:.6f}, '
                      'D_real:{:.6f}, D_fake:{:.6f}, G_l2:{:.6f}, G_class_loss:{:.6f}'.format(
                    d_loss.data.item(), g_loss.data.item(),
                    real_scores.data.mean(), fake_scores.data.mean(),
                    g_loss_wta.data.item(), g_loss_classfy.data.item()))
                history.log((epoch, cnt),
                            D_loss=d_loss,
                            g_traj_loss=g_loss_wta,
                            g_class_loss=g_loss_classfy
                            )

                # 可视化
                with canvas:
                    canvas.draw_plot(history["D_loss"])
                    canvas.draw_plot(history["g_traj_loss"])
                    canvas.draw_plot(history["g_class_loss"])
                canvas.save(os.path.join('./savepoint/', "training_progress.png"))
                # valid(centres, num_class, model_G, model_D, img, GRU_num_layers, contex_cnn_out_size, history_val, canvas_val, epoch)

        total_G_loss_wta /= cnts
        total_G_loss_class /= cnts
        with open('./savepoint/records.txt', 'a') as f:
            f.write('Epoch[{}], wta_loss each epoch:{:.3f}'.format(epoch, total_G_loss_wta))
            f.write('Epoch[{}], class_loss each epoch:{:.3f}\n'.format(epoch, total_G_loss_class))
        print('Epoch[{}], wta_loss each epoch:{:.3f}\n'.format(epoch, total_G_loss_wta))
        print('Epoch[{}], class_loss each epoch:{:.3f}\n'.format(epoch, total_G_loss_class))
        torch.save(model_G.state_dict(), './savepoint/generator_temp_{:d}.pth'.format(epoch))
        torch.save(model_D.state_dict(), './savepoint/discriminator_temp_{:d}.pth'.format(epoch))
        # print("classfy_pred", classfy_pred.shape)
        # print("traj_pred", traj_pred.shape)
        # [n, 30], [n, 2, 12]

    torch.save(model_G.state_dict(), './savepoint/generator.pth')
    torch.save(model_D.state_dict(), './savepoint/discriminator.pth')


if __name__ == "__main__":
    loss_batch = 0
    batch_count = 0
    if not torch.cuda.is_available():
        raise Exception("GPU is not available")
    model_G = Generator_net(GCN_inp_d, GCN_hid1_d, GCN_hid2_d, GCN_hid3_d, GCN_hid4_d, GCN_out_d,
                            n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, mem_len, ext_len,
                            txp_inseq, txp_outseq,
                            contex_cnn_out_size,
                            classfy_cnn_input_size, num_class,
                            tcn_input_channel, tcn_channels).cuda()
    model_D = Discriminator_net(GRU_input_size, GRU_hidden_size, GRU_num_layers).cuda()
    if load_pretrained == True:
        model_G_param = torch.load('./savepoint/generator_temp_{:d}.pth'.format(G_param))
        model_D_param = torch.load('./savepoint/discriminator_temp_{:d}.pth'.format(D_param))
        model_G.load_state_dict(model_G_param)
        model_D.load_state_dict(model_D_param)
    train(batch_count=batch_count, centres=centres, num_class=num_class, model_G=model_G,
          model_D=model_D, img=img, GRU_num_layers=GRU_num_layers,
          contex_cnn_out_size=contex_cnn_out_size)
