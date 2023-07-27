# %%

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import os
import numpy as np
import random


# %%

def walkFile(file):
    for root, dirs, files in os.walk(file):
        return root, dirs, files


config_file = 'configs/ocor/ocor_swin_large_patch4_window7_fpn_300_proposals.py'
checkpoint_file = "E:/BaiduNetdiskDownload/model.pth"

# %%

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# %%

_, _, files = walkFile('E:/Datasets/ASSR/images/test/')
# print(files)


colors = [[61, 87, 234], [99, 192, 251], [188, 176, 100], [153, 102, 68], [119, 85, 8]]
gray_colors = [255, 229, 204, 178, 153]

k = 0
for file in files:

    result = inference_detector(model, 'E:/Datasets/ASSR/images/test/' + file)

    print(k)
    #

    black_img = np.zeros([480, 640], dtype=np.uint8)
    color_img = np.zeros([480, 640, 3], dtype=np.uint8)

    score = 0
    mask = None

    # for i in range(len(result[0])-1, -1, -1):
    for i in range(len(result[0])):
        for j in range(len(result[0][i])):
            # print(result[0][i][j][4])
            if result[0][i][j][4] > 0.28:
                mask = result[1][i][j]
                score = result[0][i][j][4]
                black_img[mask] = gray_colors[i]
                color_img[mask] = colors[i]

    cv2.imwrite('E:/Datasets/ASSR/results/final/' + file[:-4] + '.png', color_img)
    cv2.imwrite('E:/Datasets/ASSR/results/final-black/' + file[:-4] + '.png', black_img)

    k = k + 1