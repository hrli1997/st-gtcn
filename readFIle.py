from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_crowds import load_crowds
import torch
import os
import math
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
from File_name import filename, file_path, h_path# , val_file_path
from readImg import centres, img


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)  # 去除首尾空格，并以\t分割为列表
            line = [float(i) for i in line]  # 数字转为float
            data.append(line)
    return np.asarray(data)
    # 返回的是二维数组，格式与原文件相符


def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # print('traj_cam', traj_cam.shape)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    return traj_uvz[:, :2].astype(int)


def image2world(traj_w, H):
    # traj.shape:[n, 2]
    traj_uvz = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # [n, 3]
    traj_cam = np.matmul(H, traj_uvz)
    traj_homog = np.transpose(traj_cam / traj_cam[2])
    return traj_homog[:, :2].astype(float)


def ReadData(OPENTRAJ_ROOT):
    # fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
    traj_dataset = load_eth(os.path.join(OPENTRAJ_ROOT, file_path))
    trajs = traj_dataset.get_trajectories()
    return traj_dataset, trajs


def ReadData_val(OPENTRAJ_ROOT):
    traj_dataset = load_eth(os.path.join(OPENTRAJ_ROOT, val_file_path))
    trajs = traj_dataset.get_trajectories()
    return traj_dataset, trajs


def Traject2Arry(full_data, H):
    H_inv = np.linalg.inv(H)
    full_data = traj_dataset.get_entries()
    # print(data.columns)
    # ndex(['frame_id', 'agent_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'scene_id',
    # 'label', 'timestamp'],dtype='object')
    frame_pedes_loc_array = full_data.values[:, :4].astype('float64')
    # print('1', frame_pedes_loc_array)
    input_array = frame_pedes_loc_array[:, 2:4]
    # print(input_array)
    frame_pedes_loc_array[:, 2:4] = world2image(input_array, H_inv)
    # temp = image2world(frame_pedes_loc_array[:, 2:4], H)
    # print(temp)
    # world2image input 1: Tx2 numpy array

    # print('2', frame_pedes_loc_array)
    # (8908, 4) <class 'numpy.ndarray'>
    return frame_pedes_loc_array


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def endpoint_calssify(traj_arry, centers_array):
    # [1, n, 2, 12] [5, 6, 2]
    a = traj_arry[0, :, :, -1]
    a.squeeze_()
    # print(a)
    # print(a.shape)
    # [5, 2]
    endclass_array = torch.zeros(a.shape[0], 1).cuda()
    per_height = centers_array[1, 0, 0] - centers_array[0, 0, 0]
    per_width = centers_array[0, 1, 1] - centers_array[0, 0, 1]
    # print('centers_array', centers_array)
    # print(per_height, per_width)
    for i, loc in enumerate(a):
        if filename == 'eth' or filename == 'htl':
            y = loc[0] // per_height
            x = loc[1] // per_width
        else:
            y = loc[1] // per_height
            x = loc[0] // per_width
        if x == centers_array.shape[1]:
            x = centers_array.shape[1] - 1
        elif x > centers_array.shape[1] or x < 0:
            print(traj_arry)
            raise Exception("width location error:{:.3f}".format(x))
        if y == centers_array.shape[0]:
            y = centers_array.shape[0] - 1
        elif y > centers_array.shape[0] or y < 0:
            print(traj_arry)
            raise Exception("height location error:{:.3f}, per_height:{:.3f}".format(y, per_height))
        # print("x, y", x, y)
        endclass_array[i] = y * centers_array.shape[1] + x
    return endclass_array





def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    # seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        # 遍历时间步
        step_ = seq_[:, :, s]
        # step_rel = seq_rel[:, :, s]
        # 取同一时间步
        for h in range(len(step_)):
            # 同一时间步中 遍历人id
            V[s, h, :] = step_[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_[h], step_[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)
    # 输出V维度：【时间（20），行人数量，2】
    # 输出A维度：【时间（20）， 行人数量，行人数量】


class Arry2Graph(Dataset):

    def __init__(self, traject_data, obs_len=8, pred_len=12, skip=1, threshold=0.002,
                min_ped=1, delim='\t', norm_lap_matr=True):
        super(Arry2Graph, self).__init__()

        self.max_peds_in_frame = 0
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        non_linear_ped = []

        frames = np.unique(traject_data[:, 0]).tolist()  # 筛选出所有出现的帧
        frame_data = []
        for frame in frames:
            frame_data.append(traject_data[frame == traject_data[:, 0], :])
        # 按帧排列，帧数为frame_data的第一维度

        num_sequences = int(
            math.ceil((len(frames) - self.seq_len + 1) / skip))
        # 序列总数 = 全部帧数-（每个序列的帧数）+1

        for idx in range(0, num_sequences * self.skip + 1, skip):
            # 对每一序列（20帧）进行处理
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + self.seq_len], axis=0)
            # 对于从目前帧起到seq_len长度的帧进行合并
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            # 将这20帧中所有行人id列出
            self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
            # 计算每帧最大行人数量
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            # 创建相对轨迹矩阵 维度为[行人数量，2(x, y坐标)，序列长度]
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            # 绝对轨迹矩阵，与上面相同
            curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                       self.seq_len))
            # 创建损失函数矩阵，维度为[行人数量，序列长度]

            num_peds_considered = 0
            _non_linear_ped = []


            for _, ped_id in enumerate(peds_in_curr_seq):
                # 挑选出每一个行人id
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                             ped_id, :]
                # 挑选出与当前行人id匹配的行组成矩阵
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                # 圆整小数点后四位
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                # 该行人出现的第一帧减目前的帧
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                # 该行人最后一帧减去目前帧
                if curr_ped_seq.shape[0] == 19:
                    pass
                if pad_end - pad_front != self.seq_len:
                    # 若在该序列中该型人出现的帧数不满足要求则转而处理下一个行人id
                    # print(pad_front, pad_end)
                    continue
                if curr_ped_seq.shape[0] != 20:
                    print("dismatch:{:f}, {:f}".format(pad_front, pad_end))
                    continue
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                # 将矩阵转换为 [2, n]，行为x/y坐标，列为序列长度（20）
                _idx = num_peds_considered
                # idx 赋0 标志行人id(在这20帧之内)
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                # curr_seq [行人序号， 2， seq], 即给x号行人pad_front到pad_end的时间内赋予坐标值

                # Linear vs Non-Linear Trajectory
                _non_linear_ped.append(
                    poly_fit(curr_ped_seq, pred_len, threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                num_peds_considered += 1

            if num_peds_considered > min_ped:
                # 若该序列中合格行人数量大于最小量
                non_linear_ped += _non_linear_ped
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_list.append(curr_seq[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        # 过去轨迹
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        # 预测轨迹
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        # 损失掩码
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        # 非线性行人轨迹

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # 第n个序列起始的行人序号列表
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_, a_ = seq_to_graph(self.obs_traj[start:end, :],  self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(self.pred_traj[start:end, :],  self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out



obs_seq_len = 8
pred_seq_len = 12

if filename == "students03":
    students_annot = os.path.join('./', 'datasets/UCY/students03/annotation.vsp')
    students_H_file = os.path.join('./', 'datasets/UCY/students03/H.txt')
    traj_dataset = load_crowds(students_annot, use_kalman=False, homog_file=students_H_file)
    H = (np.loadtxt(os.path.join("./", h_path)))
    array = Traject2Arry(traj_dataset, H)
    array[:, 2] += img.shape[3] / 2
    array[:, 3] += img.shape[2] / 2

elif filename == 'zara01':
    zara_annot = os.path.join('./', 'datasets/UCY/zara01/annotation.vsp')
    zara_H_file = os.path.join('./', 'datasets/UCY/zara01/H.txt')
    traj_dataset = load_crowds(zara_annot, use_kalman=False, homog_file=zara_H_file)
    H = (np.loadtxt(os.path.join("./", h_path)))
    array = Traject2Arry(traj_dataset, H)
    array[:, 2] += img.shape[3] / 2
    array[:, 3] += img.shape[2] / 2 # + 190
elif filename == "zara02":
    zara_annot = os.path.join('./', 'datasets/UCY/zara02/annotation.vsp')
    zara_H_file = os.path.join('./', 'datasets/UCY/zara02/H.txt')
    traj_dataset = load_crowds(zara_annot, use_kalman=False, homog_file=zara_H_file)
    H = (np.loadtxt(os.path.join("./", h_path)))
    array = Traject2Arry(traj_dataset, H)
    array[:, 2] += img.shape[3] / 2
    array[:, 3] += img.shape[2] / 2 - 50
elif filename == "students01":
    raise Exception("can't process this dataset")
else:
    traj_dataset, trajs = ReadData("./")
    H = (np.loadtxt(os.path.join("./", h_path)))
    array = Traject2Arry(traj_dataset, H)

print(array.shape)
print("max x", max(array[:, 2]))
print("min x", min(array[:, 2]))
print("max y", max(array[:, 3]))
print("min y", min(array[:, 3]))


if filename == 'zara01':
    array_train = array

elif filename == 'zara02':
    array_train = array
    '''
    一个不好的例子：
    array_train1 = array[:2600]
    array_train2 = array[3550:6050]
    array_train3 = array[7000:]
    array_val = array[2600:3550]
    array_test = array[6050:7000]
    '''
elif filename == 'students03':
    array_train = array
else:
    array_train = array
# print('array_train', array_train.shape)
# print('array_val', array_val.shape)
# 加载训练数据
'''
dset_test = Arry2Graph(
    array_test,
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1, norm_lap_matr=True)

loader_test = DataLoader(
    dset_test,
    batch_size=1,  # This is irrelative to the args batch size parameter
    shuffle=False,
    num_workers=0)
'''
dset_train = Arry2Graph(
    array_train,
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1, norm_lap_matr=True)
loader_train = DataLoader(
    dset_train,
    batch_size=1,  # This is irrelative to the args batch size parameter
    shuffle=True,
    num_workers=0)
# print(type(loader_train))
'''
dset_val = Arry2Graph(
    array_val,
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1, norm_lap_matr=True)
loader_val = DataLoader(
    dset_val,
    batch_size=1,  # This is irrelative to the args batch size parameter
    shuffle=True,
    num_workers=0)
'''














