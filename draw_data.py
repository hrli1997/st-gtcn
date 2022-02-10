import matplotlib.pyplot as plt
import numpy as np
from readFIle import loader_train, loader_val, loader_test
import time
from File_name import img_path, reference_path, filename
import cv2

img_path = reference_path
img = cv2.imread(img_path)
if filename == 'zara01' or filename == 'zara02' or filename == 'zara03' or filename == 'students01' or \
    filename == 'students03':
    loc_mode = 1
elif filename == 'eth' or filename == 'htl':
    loc_mode = 2
else:
    raise Exception("don't have that dataset")

for cnt, batch in enumerate(loader_train):
    if cnt % 1 == 0:
        obs_traj, pred_traj_gt, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch
        colors = np.random.rand(pred_traj_gt.shape[1], 3)
        # print("pred_traj", pred_traj_gt.shape, pred_traj_gt)
        # [1, n, 2, 12]
        obs_traj = obs_traj.squeeze().permute(0, 2, 1)
        obs_traj = obs_traj.numpy()
        pred_traj_gt = pred_traj_gt.squeeze().permute(0, 2, 1)
        pred_traj_gt = pred_traj_gt.numpy()
        # [n, seq, 2]
        # print(obs_traj, obs_traj.shape)
        # exit()
        '''
        for i, seq in enumerate(obs_traj):
            color = colors[i]
            for j, loc in enumerate(seq):
                x, y = loc
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
                # ax.scatter(x, y, s=30, color=color)
                # time.sleep(0.2)
            plt.show()
        for i, seq in enumerate(pred_traj_gt):
            color = colors[i]
            for j, loc in enumerate(seq):
                x, y = loc
                ax.scatter(x, y, s=30, color=color)
                # time.sleep(0.2)
            plt.show()
        '''
        for i, trajs in enumerate(zip(obs_traj, pred_traj_gt)):
            obs, pred = trajs
            x1, y1 = -1, -1
            for loc in obs:
                if x1 == -1 or y1 == -1:
                    if loc_mode == 1:
                        x1, y1 = loc
                    elif loc_mode == 2:
                        y1, x1 = loc
                    y1 = int(img.shape[0] - y1)
                    continue
                if loc_mode == 1:
                    x2, y2 = loc
                elif loc_mode == 2:
                    y2, x2 = loc
                y2 = int(img.shape[0] - y2)
                cv2.line(img, (x1, y1), (x2, y2), color=(colors[i]*255), thickness=2)
                x1, y1 = x2, y2
            for loc in pred:
                if loc_mode == 1:
                    x2, y2 = loc
                elif loc_mode == 2:
                    y2, x2 = loc
                y2 = int(img.shape[0] - y2)
                cv2.line(img, (x1, y1), (x2, y2), color=(colors[i]*255),  thickness=2)
                x1, y1 = x2, y2
            cv2.imshow("image-lines", img)
            cv2.waitKey(10)
        cv2.imwrite("traj_pic.jpg", img)





