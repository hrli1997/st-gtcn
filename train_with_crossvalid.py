from Generator import Generator_net, single_output_train
from readFIle import loader_train, endpoint_calssify, dset_train#, loader_test
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
import random
from threading import Thread
from multiprocessing import Process
import os

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

num_epoch = 500
batch_size = 4

# 是否加载预训练模型
load_pretrained = False
'''
D_param = 249  # last valid epoch:27
G_param = D_param
'''
single_output_train = single_output_train
cross_valid_num = 6
threads = 3

D_param_pro1 = 199
G_param_pro1 = D_param_pro1
D_param_pro2 = 199
G_param_pro2 = D_param_pro2
D_param_pro3 = 199
G_param_pro3 = D_param_pro3


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


def cross_valid(k, k_temp, centres, num_class, model_G, model_D, img, GRU_num_layers,
                contex_cnn_out_size, batch_size):
    '''
    :param centres:
    :param num_class:
    :param model_G:
    :param model_D:
    :param img:
    :param GRU_num_layers:
    :param contex_cnn_out_size:
    :param batch_size:
    :param k: 共多少折(1~n)
    :param k_temp: 目前第几折(1~k)
    :return: none
    '''

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
    index_train = index_full
    index_train[sample_per_part * (k_temp - 1): sample_per_part * k_temp] = []

    # 打乱train part 顺序
    random.seed(time.time())
    for i in range(len(index_train) // 2):
        a = random.randint(0, len(index_train) - 1)
        b = random.randint(0, len(index_train) - 1)
        i = index_train[a]
        index_train[a] = index_train[b]
        index_train[b] = i
    # print(index_test)
    # print(index_train)
    '''
    ********************数据（索引）处理完毕******************
    '''
    '''
    开始训练
    '''
    history = hl.History()
    criterion_D = nn.BCELoss().cuda()  # 单目标二分类交叉熵函数
    d_optimizer = torch.optim.Adam(model_D.parameters(), lr=0.001)
    g_optimizer = torch.optim.Adam(model_G.parameters(), lr=0.0001, weight_decay=1e-8)
    # g_optimizer = torch.optim.SGD(model_G.parameters(), lr=1e-9)#, weight_decay=1e-8)

    global num_epoch
    model_G.train()
    model_D.train()
    for epoch in range(num_epoch):  # 进行多个epoch的训练
        total_G_loss_wta = 0
        total_G_loss_class = 0
        total_D_loss = 0
        cnts = 0
        batch_loss_D = 0
        batch_loss_D_real = 0
        batch_loss_D_fake = 0
        batch_loss_G = 0
        batch_loss_G_class = 0
        batch_loss_G_traj = 0
        time_start = time.time()
        cnt = -1
        for index in index_train:
            batch = dset_train.__getitem__(index)
            cnt += 1
            # Get data
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
            classfy_pred.detach_()
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
            d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失

            batch_loss_D = batch_loss_D + d_loss
            batch_loss_D_fake += d_loss_fake
            batch_loss_D_real += d_loss_real
            if (cnt + 1) % batch_size == 0:
                # 损失函数和优化
                batch_loss_D = batch_loss_D / batch_size
                d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
                batch_loss_D.backward()  # 将误差反向传播
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

            if single_output_train == 0:  # 轨迹+分类
                g_loss = g_loss_traj + g_loss_classfy
            elif single_output_train == 1:  # 训练轨迹
                g_loss = g_loss_traj
            elif single_output_train == 2:  # 训练分类
                g_loss = g_loss_classfy
            batch_loss_G = batch_loss_G + g_loss
            batch_loss_G_class = batch_loss_G_class + g_loss_classfy
            batch_loss_G_traj = batch_loss_G_traj + g_loss_wta
            if (cnt + 1) % batch_size == 0:
                # bp and optimize
                batch_loss_G = batch_loss_G / batch_size
                g_optimizer.zero_grad()  # 梯度归0
                torch.autograd.set_detect_anomaly(True)
                batch_loss_G.backward()  # 进行反向传播
                g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

            if (cnt + 1) % batch_size == 0:
                batch_loss_G_traj = batch_loss_G_traj / batch_size
                batch_loss_G_class = batch_loss_G_class / batch_size
                batch_loss_D_real /= batch_size
                batch_loss_D_fake /= batch_size
                time_end = time.time()

                print('Batch/Epoch[{}/{}], used_time:{:.3f}, cross:{:d}'.format(
                    cnt, epoch, time_end - time_start, k_temp
                ))

                time_start = time.time()

                print('d_loss:{:.6f}, g_loss:{:.6f}, '
                      'd_loss_real:{:.6f}, d_loss_fake:{:.6f}'
                      'G_l2:{:.6f}, G_class_loss:{:.6f}'.format(
                    batch_loss_D.data.item(), batch_loss_G.data.item(),
                    batch_loss_D_real, batch_loss_D_fake,
                    batch_loss_G_traj.data.item(), batch_loss_G_class.data.item()))

                history.log((epoch, cnt),
                            D_loss=batch_loss_D,
                            g_traj_loss=batch_loss_G_traj,
                            g_class_loss=batch_loss_G_class)
                # 可视化
                '''
                with canvas:
                    canvas.draw_plot(history["D_loss"])
                    canvas.draw_plot(history["g_traj_loss"])
                    canvas.draw_plot(history["g_class_loss"])
                canvas.save(os.path.join('./savepoint/', "training_progress.png"))
                '''

                # btchloss归零
                batch_loss_G = 0
                batch_loss_G_class = 0
                batch_loss_G_traj = 0
                batch_loss_D = 0
                batch_loss_D_real = 0
                batch_loss_D_fake = 0

            total_G_loss_wta += g_loss_wta.detach()
            total_G_loss_class += g_loss_classfy.detach()
            total_D_loss += d_loss.detach()
            cnts = cnt
        total_G_loss_wta /= cnts
        total_G_loss_class /= cnts
        total_D_loss /= cnts
        with open('./savepoint/records_cross{:d}.txt'.format(k_temp), 'a') as f:
            f.write('Epoch[{}], wta_loss each epoch:{:.3f}'.format(epoch, total_G_loss_wta))
            f.write('Epoch[{}], class_loss each epoch:{:.3f}\n'.format(epoch, total_G_loss_class))
            f.write('Epoch[{}], deiscrim_loss each epoch:{:.3f}\n'.format(epoch, total_D_loss))
        # print('Cross{:d}:Epoch[{}], wta_loss each epoch:{:.3f}\n'.format(k_temp, epoch, total_G_loss_wta))
        # print('Cross{:d}:Epoch[{}], class_loss each epoch:{:.3f}\n'.format(k_temp, epoch, total_G_loss_class))
        if (epoch + 1) % 50 == 0:
            torch.save(model_G.state_dict(), './savepoint/generator_temp_cross{:d}_{:d}.pth'.format(k_temp, epoch))
            torch.save(model_D.state_dict(), './savepoint/discriminator_temp_cross{:d}_{:d}.pth'.format(k_temp, epoch))


class myProcess(Process):
    def __init__(self, processID, name, k, k_temp, centres, num_class, img, GRU_num_layers,
                 contex_cnn_out_size, batch_size, GCN_inp_d, GCN_hid1_d, GCN_hid2_d, GCN_out_d,
                 n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, mem_len, ext_len,
                 txp_inseq, txp_outseq,
                 classfy_cnn_input_size,
                 tcn_input_channel, tcn_channels,
                 GRU_input_size, GRU_hidden_size):
        super(myProcess, self).__init__()
        self.processID = processID
        self.name = name
        self.k = k
        self.k_temp = k_temp
        self.centres = centres
        self.num_class = num_class
        self.img = img
        self.GRU_num_layers = GRU_num_layers
        self.contex_cnn_out_size = contex_cnn_out_size
        self.batch_size = batch_size
        self.GCN_inp_d = GCN_inp_d
        self.GCN_hid1_d = GCN_hid1_d
        self.GCN_hid2_d = GCN_hid2_d
        self.GCN_out_d = GCN_out_d
        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.txp_inseq = txp_inseq
        self.txp_outseq = txp_outseq
        self.classfy_cnn_input_size = classfy_cnn_input_size
        self.tcn_input_channel = tcn_input_channel
        self.tcn_channels = tcn_channels
        self.GRU_hidden_size = GRU_hidden_size
        self.GRU_input_size = GRU_input_size

    def run(self):
        print("开始进程：" + self.name)
        model_G = Generator_net(self.GCN_inp_d, self.GCN_hid1_d, self.GCN_hid2_d, self.GCN_out_d,
                                self.n_token, self.n_layer, self.n_head, self.d_model, self.d_head,
                                self.d_inner, self.dropout, self.mem_len, self.ext_len,
                                self.txp_inseq, self.txp_outseq,
                                self.contex_cnn_out_size,
                                self.classfy_cnn_input_size, self.num_class,
                                self.tcn_input_channel, self.tcn_channels)
        model_D = Discriminator_net(self.GRU_input_size, self.GRU_hidden_size, self.GRU_num_layers)
        torch.cuda.set_device(1)
        if not torch.cuda.is_available():
            raise Exception("GPU is not available")
        model_G = model_G.cuda()
        model_D = model_D.cuda()
        if load_pretrained:
            if self.processID == 1:
                G_param = G_param_pro1
                D_param = D_param_pro1
            elif self.processID == 2:
                G_param = G_param_pro2
                D_param = D_param_pro2
            elif self.processID == 3:
                G_param = G_param_pro3
                D_param = D_param_pro3
            else:
                raise RuntimeError("process ID wrong:{:d}".format(self.processID))
            model_G_param = torch.load('./savepoint/generator_temp_cross{:d}_{:d}.pth'.
                                       format(self.k_temp, G_param))
            model_D_param = torch.load('./savepoint/discriminator_temp_cross{:d}_{:d}.pth'
                                       .format(self.k_temp, D_param))
            model_G.load_state_dict(model_G_param)
            model_D.load_state_dict(model_D_param)
        cross_valid(self.k, self.k_temp, centres=self.centres, num_class=self.num_class,
                    model_G=model_G, model_D=model_D, img=self.img, GRU_num_layers=self.GRU_num_layers,
                    contex_cnn_out_size=self.contex_cnn_out_size, batch_size=self.batch_size)
        print("退出进程：" + self.name)


if __name__ == "__main__":

    loss_batch = 0
    '''
    gpu_num = torch.cuda.device_count()
    torch.cuda.set_device(0)
    '''

    thread1 = myProcess(1, "Process-1", k=cross_valid_num, k_temp=4, centres=centres, num_class=num_class,
                        img=img, GRU_num_layers=GRU_num_layers,
                        contex_cnn_out_size=contex_cnn_out_size, batch_size=batch_size,
                        GCN_inp_d=GCN_inp_d, GCN_hid1_d=GCN_hid1_d, GCN_hid2_d=GCN_hid2_d,
                        GCN_out_d=GCN_out_d,
                        n_token=n_token, n_layer=n_layer, n_head=n_head, d_model=d_model,
                        d_head=d_head, d_inner=d_inner, dropout=dropout, mem_len=mem_len,
                        ext_len=ext_len,
                        txp_inseq=txp_inseq, txp_outseq=txp_outseq,
                        classfy_cnn_input_size=classfy_cnn_input_size,
                        tcn_input_channel=tcn_input_channel, tcn_channels=tcn_channels,
                        GRU_input_size=GRU_input_size, GRU_hidden_size=GRU_hidden_size)
    thread1.daemon = True
    thread1.start()
    if threads >= 2:
        thread2 = myProcess(2, "Process-2", k=cross_valid_num, k_temp=5, centres=centres, num_class=num_class,
                            img=img, GRU_num_layers=GRU_num_layers,
                            contex_cnn_out_size=contex_cnn_out_size, batch_size=batch_size,
                            GCN_inp_d=GCN_inp_d, GCN_hid1_d=GCN_hid1_d, GCN_hid2_d=GCN_hid2_d,
                            GCN_out_d=GCN_out_d,
                            n_token=n_token, n_layer=n_layer, n_head=n_head, d_model=d_model,
                            d_head=d_head, d_inner=d_inner, dropout=dropout, mem_len=mem_len,
                            ext_len=ext_len,
                            txp_inseq=txp_inseq, txp_outseq=txp_outseq,
                            classfy_cnn_input_size=classfy_cnn_input_size,
                            tcn_input_channel=tcn_input_channel, tcn_channels=tcn_channels,
                            GRU_input_size=GRU_input_size, GRU_hidden_size=GRU_hidden_size
                            )
        thread2.daemon = True
        thread2.start()
    if threads >= 3:

        thread3 = myProcess(3, "Process-3", k=cross_valid_num, k_temp=6, centres=centres, num_class=num_class,
                            img=img, GRU_num_layers=GRU_num_layers,
                            contex_cnn_out_size=contex_cnn_out_size, batch_size=batch_size,
                            GCN_inp_d=GCN_inp_d, GCN_hid1_d=GCN_hid1_d, GCN_hid2_d=GCN_hid2_d,
                            GCN_out_d=GCN_out_d,
                            n_token=n_token, n_layer=n_layer, n_head=n_head, d_model=d_model,
                            d_head=d_head, d_inner=d_inner, dropout=dropout, mem_len=mem_len,
                            ext_len=ext_len,
                            txp_inseq=txp_inseq, txp_outseq=txp_outseq,
                            classfy_cnn_input_size=classfy_cnn_input_size,
                            tcn_input_channel=tcn_input_channel, tcn_channels=tcn_channels,
                            GRU_input_size=GRU_input_size, GRU_hidden_size=GRU_hidden_size
                            )
        thread3.daemon = True
        thread3.start()
    print("Main thread sleep")
    while True:
        pass
