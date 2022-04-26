import pandas as pd
import json
from tqdm import tqdm
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
from scipy.stats import skew, kurtosis
from scipy.signal import resample
import re
import os
import sklearn
import glob
import scipy.io as sio

warnings.filterwarnings('ignore')


def get_sub(PATH):
    # 读取预测文件
    FILES = os.listdir(PATH)

    SUB = np.sort([f for f in FILES if 'pred' in f])
    SUB_CSV = [pd.read_csv(PATH + k) for k in SUB]

    print('We have %i submission files...' % len(SUB))
    print();
    print(SUB)

    return SUB, SUB_CSV


SUB, SUB_CSV = get_sub('final/')
# 读取权重文件
ff = pd.read_csv('weight.csv')
threshold_list = [16.468448415811796,
                  17.81664276267562,
                  20.185599945408224,
                  21.128897793968036,
                  15.290330382363098,
                  15.299899954526154,
                  10.578932641957495,
                  11.032894420664729,
                  16.223668278875703,
                  8.5659931206263,
                  11.072673525415087,
                  17.507474051125413,
                  16.35911186689198,
                  5.26504456402904,
                  12.06633038453706,
                  16.19896696904366,
                  8.755300379278433,
                  20.815442559031585]

s = SUB_CSV[0].copy()
for f in range(18):
    s[f'pred_{f}'] *= ff[ff['f'] == f][f'params_oof_weights_0'].values
    for i in range(1, len(SUB_CSV)):
        s[f'pred_{f}'] += ff[ff['f'] == f][f'params_oof_weights_{i}'].values * SUB_CSV[i][f'pred_{f}']

lab_cols = [f'label_{i}' for i in range(18)]
pred_cols = [f'pred_{i}' for i in range(18)]


# 阈值处理
for col in lab_cols:
    s[col] = 1
s[lab_cols] = s[pred_cols].values
for i in range(18):
    s[lab_cols[i]] = s[lab_cols[i]].apply(lambda x: 1 if x > threshold_list[i] else 0)


# 后处理 互斥项优化
s['label_13'] = s[['label_15', 'label_13']].apply(lambda x: 0 if x.label_15 == 1 else x.label_13, axis=1)
s['label_13'] = s[['label_14', 'label_13']].apply(lambda x: 0 if x.label_14 == 1 else x.label_13, axis=1)


# 生成提交文件
s[lab_cols] = s[lab_cols].astype('str')
s['label'] = s['label_0'].str.cat([s[f'label_{i}'] for i in range(1, 18)], sep=',')
s[['id', 'label']].to_csv('sub_final.csv', index=False)
print(s[['id', 'label']].head())