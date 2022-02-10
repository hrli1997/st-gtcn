from readFIle import loader_train, endpoint_calssify
from GCN import GraphConv, Graph_normalize
from transformer_xl import MemTransformerLM
import torch
from time_expand_conv import TxpCnn
from tcn import TemporalConvNet
import torchvision.models as models
from torch import optim, nn
from readImg import EndPoint_Pred_CNN  # img, centres,
from contex_cnn import ContexConv

single_output_train = 0
# 0:两个输出共同更新参数
# 1:从轨迹输出更新参数
# 2:从分类输出更新参数


class Generator_net(nn.Module):

    def __init__(self, gcn_in_d, gcn_hid1_d, gcn_hid2_d, gcn_out_d,
                 n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, mem_len, ext_len,
                 txp_inseq, txp_outseq,
                 contex_cnn_out,
                 classfy_input, num_class,
                 tcn_input_channel, tcn_channels):
        super(Generator_net, self).__init__()

        # parameters define
        self.gcn_in_d = gcn_in_d
        self.gcn_hid1_d = gcn_hid1_d
        self.gcn_hid2_d = gcn_hid2_d
        self.gcn_out_d = gcn_out_d

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

        self.context_cnn_out = contex_cnn_out

        self.classfy_input = classfy_input
        self.num_class = num_class

        self.tcn_input_channel = tcn_input_channel
        self.tcn_channels = tcn_channels

        # networks define
        self.gcn = GraphConv(gcn_in_d, gcn_hid1_d, gcn_hid2_d, gcn_out_d)
        self.transformer = MemTransformerLM(n_token, n_layer, n_head, d_model, d_head, d_inner, dropout,
                                            dropatt=dropout, pre_lnorm=True, mem_len=mem_len, ext_len=ext_len)
        self.txpcnn = TxpCnn(txp_inseq, txp_outseq)
        self.context_cnn = ContexConv(contex_cnn_out)
        self.class_cnn = EndPoint_Pred_CNN(classfy_input, num_class)
        # 用于替代resnet进行终点分类

        self.tcn_net = TemporalConvNet(tcn_input_channel, tcn_channels)

    def forward(self, V_obs, A_obs, img, centres):
        '''
        :param V_obs: 顶点矩阵
        :param A_obs: 邻接矩阵
        :param img: 图像矩阵
        :param centres: 分割图像的中点矩阵
        :return: 分类网络输出、轨迹预测网络输出
        '''
        # 创建GCN输出矩阵
        gcn_full_out = torch.zeros([V_obs.shape[0], V_obs.shape[1], self.gcn_out_d]).cuda()
        frame_count = 0
        # 将graph分步输入GCN
        for A_obs_fram, V_obs_fram in zip(A_obs, V_obs):
            A_normed_frame = Graph_normalize(A_obs_fram)
            gcn_out = self.gcn(A_normed_frame, V_obs_fram)
            # print("gcn_out", gcn_out.shape)
            # [n, 10]
            gcn_full_out[frame_count] = gcn_out
            frame_count = frame_count + 1
        # 将语义图输入CNN，获取语义编码信息
        context = self.context_cnn(img.cuda())
        context.squeeze_()
        # 将图像信息复制扩展，变为与图信息相等形状
        context_expand = torch.zeros(gcn_full_out.shape[0], gcn_full_out.shape[1], context.shape[0]).cuda()
        for i in range(context_expand.shape[0]):
            for j in range(context_expand.shape[1]):
                context_expand[i, j, :] = context
        # 将两个矩阵拼接
        features_full_out = torch.cat([gcn_full_out, context_expand], dim=2)
        # 创建transformer输出换存
        trans_full_out = torch.zeros([features_full_out.shape[0], features_full_out.shape[1], self.d_model]).cuda()
        seg_num = features_full_out.shape[0] // self.mem_len
        # 创建记忆缓存
        mems = tuple()
        # 按记忆长度（4）切割，分批送入transformer
        for i in range(seg_num):
            inp = features_full_out[self.mem_len * i: self.mem_len * (i + 1)]
            out = self.transformer(inp, *mems)
            trans_full_out[self.mem_len * i: self.mem_len * (i + 1)] = out[0]
            mems = out[1:]
        trans_full_out.unsqueeze_(0)
        '''
            至此编码部分完成，接下来是解码部分
        '''
        # 扩展时间步，将8步扩展为12步
        timexpanded_out = self.txpcnn(trans_full_out)
        '''
            以下部分是分类网络部分
        '''
        # 调换维度，准备送入分类网络
        classfy_input = timexpanded_out.permute(2, 0, 1, 3)
        # 获得分类输出
        classfy_output = self.class_cnn(classfy_input)
        '''
            以下部分是具体轨迹生成部分
        '''
        # 将分类网络输出的logsofmax信息转化为坐标信息
        classfy_output_location = torch.zeros(classfy_output.shape[0], 2).cuda()
        '''
        修改1：将分类网络传过来的数据detach，这样在计算详细轨迹梯度时不会更新分类网络
        '''
        if single_output_train == 0:
            classfy_arry = classfy_output.cpu().detach()
        elif single_output_train == 1:
            classfy_arry = classfy_output.cpu()
        elif single_output_train == 2:
            classfy_arry = classfy_output.cpu().detach()
        else:
            raise Exception("output mode not support")

        for i, loc in enumerate(classfy_arry):
            indx = loc.argmax()
            y = indx // centres.shape[1]
            x = indx - y * centres.shape[1]
            # print(x, y, i)
            classfy_output_location[i] = centres[y, x]
        # 将得到的坐标信息扩展，使其符合transformer输出的形状
        classfy_output_location_expand = torch.zeros(timexpanded_out.shape[0],
                                                     timexpanded_out.shape[1],
                                                     classfy_output_location.shape[0],
                                                     classfy_output_location.shape[1]).cuda()
        for num1, i in enumerate(classfy_output_location_expand):
            for num2, _ in enumerate(i):
                classfy_output_location_expand[num1, num2] = classfy_output_location
        # 将分类信息与transformer信息拼接
        trajrct_pred_input = torch.cat([timexpanded_out, classfy_output_location_expand], dim=3).cuda()
        # 将得到的矩阵形状转化以符合tcn输入
        trajrct_pred_input.squeeze_()
        trajrct_pred_input = trajrct_pred_input.permute(1, 2, 0)
        # 输入tcn网络获得最终的输出
        tcn_output = self.tcn_net(trajrct_pred_input)

        return classfy_output, tcn_output, context
