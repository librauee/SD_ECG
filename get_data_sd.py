import os
import itertools
import time
import random
import glob
import math
import logging

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from scipy import signal


# 数据预处理方式一
def get_data(path):
    ecg_data = pd.read_csv(path)
    l = len(ecg_data)
    if l < 16001:
        # 填充数据长度
        leads = [i for i in ecg_data.columns if 'LEAD' in i]
        ecg_data = ecg_data[leads].values.T
        ecg_data = np.pad(ecg_data, ((0, 0), (0, 16001 - l)), )
    else:
        # 截断长数据
        leads = [i for i in ecg_data.columns if 'LEAD' in i]
        ecg_data = ecg_data[leads].values.T[:, :16001]

    # 保存成矩阵形式
    mdic = {"ecg_data": ecg_data}
    sio.savemat(path.replace('ecg_data', 'ecg_data_mat').replace('csv', 'mat'), mdic)

# 预处理所有csv文件
train_path = glob.glob('ecg_data/*.csv')
train_path.sort()
data_dict = {}
for path in tqdm(train_path):
    ecg_data = get_data(path)


#数据预处理方式二
def get_data(path):
    ecg_data = pd.read_csv(path)
    leads = [i for i in ecg_data.columns if 'LEAD' in i]
    ecg_data = ecg_data[leads].values.T
    # 重采样
    ecg_data = signal.resample(ecg_data, 6666, axis=1)
    return ecg_data


train_path = glob.glob('ecg_data/*.csv')
train_path.sort()
data_dict = {}
for path in tqdm(train_path):
    ecg_data = get_data(path)
    data_dict[os.path.basename(path)[:-4]] = ecg_data

np.save('data.npy', data_dict)

