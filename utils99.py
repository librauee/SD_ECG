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
                                      ExponentialLR,
                                      ReduceLROnPlateau
                                      )

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import normalize
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from scipy import signal
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import Normalizer, MinMaxScaler


def fix_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_logger(output_dir, version):

    log_file = output_dir + f"v{version}.log"
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def transform(v):

    v= (v - v.mean(axis=0).reshape((1, v.shape[1]))) / (v.max(axis=0).reshape((1, v.shape[1]))
                                                        -v.min(axis=0).reshape((1, v.shape[1])) + 1e-6)
    return v

def mixup(x, y, alpha=0.5):

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        pt = pt.clamp(min=0.00001, max=0.99999)

        # eps = 1e-9
        # pt = torch.clamp(pt, eps, 1.0 - eps)

        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def cal_weight(CONFIG):

    labels = pd.read_csv('label_and_example/train_label_1217.csv')
    labels[[f'label_{i}' for i in range(18)]] = labels.label.str.split(',', expand=True)
    lab_cols = [i for i in labels.columns if i not in ['id', 'label']]
    labels[lab_cols] = labels[lab_cols].astype('int')
    labels = labels[lab_cols].values
    sum_1 = np.sum(labels, axis=0)

    if CONFIG.weight == 'base':
        return len(labels) / sum_1
    elif CONFIG.weight == 'log_base':
        return 1. / np.log(sum_1 + 1)
    elif CONFIG.weight == '1_score':
        return 1. / np.array([.6, .8, .9, .9, .7, .8, .5, .5, .9, .6, .9, .7, .6, .1, .8, .8, .4, .8])
    elif CONFIG.weight == 'norm_1_score':
        weight = 1. / np.array([.6, .8, .9, .9, .7, .8, .5, .5, .9, .6, .9, .7, .6, .1, .8, .8, .4, .8])
        return weight / sum(weight)
    elif CONFIG.weight == 'norm_1_log':
        weight = 1. / np.log(sum_1 + 1)
        return weight / sum(weight)
    else:
        raise ValueError('no this weight function', CONFIG.weight)

class WeightedMultilabel(nn.Module):

    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()

class MultiLabelCircleLoss(nn.Module):

    def __init__(self, reduction="mean", inf=1e12):
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf

    def forward(self, logits, labels):
        logits = (1 - 2 * labels) * logits              # <3, 4>
        logits_neg = logits - labels * self.inf         # <3, 4>
        logits_pos = logits - (1 - labels) * self.inf   # <3, 4>
        zeros = torch.zeros_like(logits[..., :1])       # <3, 1>
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <3, 5>
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)  # <3, 5>
        neg_loss = torch.logsumexp(logits_neg, dim=-1)       # <3, >
        pos_loss = torch.logsumexp(logits_pos, dim=-1)       # <3, >
        loss = neg_loss + pos_loss
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class f1_loss(nn.Module):

    def __init__(self):
        super(f1_loss, self).__init__()

    def forward(self, predict, target):
        loss = 0
        lack_cls = target.sum(dim=0) == 0
        if lack_cls.any():
            loss += F.binary_cross_entropy_with_logits(
                predict[:, lack_cls], target[:, lack_cls])
        predict = torch.sigmoid(predict)
        predict = torch.clamp(predict * (1 - target), min=0.01) + predict * target
        tp = predict * target
        tp = tp.sum(dim=0)
        precision = tp / (predict.sum(dim=0) + 1e-8)
        recall = tp / (target.sum(dim=0) + 1e-8)
        f1 = 2 * (precision * recall / (precision + recall + 1e-8))
        return 1 - f1.mean() + loss


class Trainer:
    def __init__(self, net, CONFIG, LOGGER, labels, test_path=None, fold=0):

        self.test_path = test_path
        self.CONFIG = CONFIG
        if self.CONFIG.sample:
            self.data_dict = np.load('data.npy', allow_pickle=True).item()

        self.LOGGER = LOGGER
        self.labels = labels
        self.num_epochs = self.CONFIG.num_epochs
        self.net = net.to(self.CONFIG.device)
        self.batch_size = self.CONFIG.batch_size
        self.fold = fold
        self.early_stop_count = 0
        self.early_stop_metric = CONFIG.metric
        self.best_score = 0
        self.best_loss = float('inf')
        self.metric = self.CONFIG.metric

        # criterion
        if CONFIG.loss == 'WeightedMultilabel':
            self.weight = torch.tensor(cal_weight(CONFIG), dtype=torch.float).to(self.CONFIG.device)
            self.criterion = WeightedMultilabel(self.weight)
        elif CONFIG.loss == 'BCEFocalLoss':
            self.criterion = BCEFocalLoss()
        elif CONFIG.loss == 'WeightedBCEFocalLoss':
            self.weight = torch.tensor(cal_weight(CONFIG), dtype=torch.float).to(self.CONFIG.device)
            self.criterion = BCEFocalLoss(alpha=self.weight)
        elif CONFIG.loss == 'MultiLabelCircleLoss':
            self.criterion = MultiLabelCircleLoss()
        elif CONFIG.loss == 'F1':
            self.criterion = f1_loss()
        else:
            raise ValueError('no this loss function', CONFIG.loss)

        # optimizer
        if CONFIG.optimizer == 'Ranger':
            from ranger import Ranger
            self.optimizer = Ranger(self.net.parameters(), lr=self.CONFIG.lr)
        elif CONFIG.optimizer == 'AdamW':
            self.optimizer = AdamW(self.net.parameters(), lr=self.CONFIG.lr)
        else:
            raise ValueError('no this optimizer', CONFIG.optimizer)

        # scheduler
        if CONFIG.scheduler == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.CONFIG.num_epochs, eta_min=5e-6)
        elif CONFIG.scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5)
        elif CONFIG.scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5,
                          verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        else:
            raise ValueError('no this scheduler', CONFIG.scheduler)

        self.ECGDataset = self.getDataset()
        self.train_loaders = self.get_dataloader('train', self.labels, self.CONFIG.fold, self.CONFIG.seed, None,
                                            self.CONFIG.batch_size, self.fold)
        self.val_loaders = self.get_dataloader('valid', self.labels, self.CONFIG.fold, self.CONFIG.seed, None,
                                            self.CONFIG.batch_size, self.fold)

    def getDataset(self):

        if self.CONFIG.sample:
            normalizer = Normalizer()
            class ECGDataset(Dataset):

                def __init__(self2, data, phase):

                    self2.phase = phase
                    self2.data = data

                    if self2.phase != 'test':
                        self2.lab_cols = [i for i in self2.data.columns if i not in ['id', 'label']]
                        self2.data.reset_index(inplace=True, drop=True)

                def __getitem__(self2, idx):
                    if self2.phase == 'train':
                        id_ = self2.data.iloc[idx]['id']
                        ecg_data = self.data_dict[id_]
                        # transform(ecg_data, train=True).copy()
                        if self.CONFIG.norm:
                            ecg_data = normalizer.fit_transform(ecg_data)
                        if self.CONFIG.transform:
                            ecg_data = transform(ecg_data)
                        signal = torch.FloatTensor(ecg_data)
                        target = torch.FloatTensor(self2.data.iloc[idx][self2.lab_cols])
                    elif self2.phase == 'valid':
                        id_ = self2.data.iloc[idx]['id']
                        ecg_data = self.data_dict[id_]
                        if self.CONFIG.norm:
                            ecg_data = normalizer.fit_transform(ecg_data)
                        # if self.CONFIG.transform:
                        #     ecg_data = minmax.fit_transform(ecg_data)
                        signal = torch.FloatTensor(ecg_data)
                        target = torch.FloatTensor(self2.data.iloc[idx][self2.lab_cols])

                    else:
                        id_ = self2.data[idx]
                        ecg_data = self.data_dict[id_]
                        if self.CONFIG.norm:
                            ecg_data = normalizer.fit_transform(ecg_data)
                        # if self.CONFIG.minmax:
                        #     ecg_data = minmax.fit_transform(ecg_data)
                        signal = torch.FloatTensor(ecg_data)
                        target = torch.FloatTensor([0] * 18)

                    return signal, target

                def __len__(self2):
                    return len(self2.data)


        else:


            class ECGDataset(Dataset):
                def __init__(self2, data, phase):
                    from sklearn.preprocessing import StandardScaler
                    self2.scaler = StandardScaler()
                    self2.phase = phase
                    self2.data = data
                    if self2.phase != 'test':
                        self2.lab_cols = [i for i in self2.data.columns if i not in ['id', 'label']]
                        self2.data.reset_index(inplace=True, drop=True)

                def __getitem__(self2, idx):
                    if self2.phase == 'train':
                        ecg_data = sio.loadmat('ecg_data_mat/' + self2.data.iloc[idx]['id'] + '.mat')['ecg_data']
                        if self.CONFIG.transform:
                            # ecg_data = transform(ecg_data)
                            ecg_data = self2.scaler.fit_transform(ecg_data)
                        signal = torch.FloatTensor(ecg_data)
                        target = torch.FloatTensor(self2.data.iloc[idx][self2.lab_cols])
                    elif self2.phase == 'valid':
                        ecg_data = sio.loadmat('ecg_data_mat/' + self2.data.iloc[idx]['id'] + '.mat')['ecg_data']
                        if self.CONFIG.transform:
                            # ecg_data = transform(ecg_data)
                            ecg_data = self2.scaler.fit_transform(ecg_data)
                        signal = torch.FloatTensor(ecg_data)
                        target = torch.FloatTensor(self2.data.iloc[idx][self2.lab_cols])

                    else:
                        ecg_data = sio.loadmat('ecg_data_mat/' + self2.data[idx] + '.mat')['ecg_data']
                        if self.CONFIG.transform:
                            # ecg_data = transform(ecg_data)
                            ecg_data = self2.scaler.fit_transform(ecg_data)
                        signal = torch.FloatTensor(ecg_data)
                        target = torch.FloatTensor([0] * 18)

                    return signal, target

                def __len__(self2):
                    return len(self2.data)

        return ECGDataset

    def get_dataloader(self, phase, labels, folds=5, seed=1009, test=None, batch_size=96, fold=0):

        lab_cols = [f'label_{i}' for i in range(18)]
        KF = MultilabelStratifiedKFold(folds, random_state=seed, shuffle=True)
        for fold_, (trn_idx, val_idx) in enumerate(KF.split(labels['id'].values, labels[lab_cols].values)):

            if fold == fold_:
                val = labels.iloc[val_idx]
                train = labels.iloc[trn_idx]
                break

        if self.CONFIG.pseudo:
            labels_pseudo = pd.read_pickle('pseudo.pkl')
            for fold_, (trn_idx, val_idx) in enumerate(KF.split(labels_pseudo['id'].values, labels_pseudo[lab_cols].values)):
                if fold == fold_:
                    train_pseudo = labels_pseudo.iloc[trn_idx]
                    break
            train = pd.concat([train, train_pseudo])

        if phase == 'train':
            dataset = self.ECGDataset(train, phase)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        elif phase == 'test':
            dataset = self.ECGDataset(test, phase)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size * 2, shuffle=False)
        else:
            dataset = self.ECGDataset(val, phase)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size * 2, shuffle=False)
        return dataloader



    def cal_score(self, true, pred):
        if self.CONFIG.one:

            true = true.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            for i in range(18):
                score_list.append(roc_auc_score(true[:, i], pred[:, i]))

            pred = np.where(pred > 0.5, 1, 0)
            score = f1_score(true, pred, average='macro')
            self.LOGGER.info(f'f1_score: {score}')
            self.LOGGER.info(f'auc_score_each: {score_list}')
            return score_list
        else:

            true = true.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            pred = np.where(pred > 0.5, 1, 0)
            score = f1_score(true, pred, average='macro')
            score_list = []
            for i in range(18):
                score_list.append(f1_score(true[:, i], pred[:, i]))

            self.LOGGER.info(f'f1_score: {score}')
            self.LOGGER.info(f'f1_score_each: {score_list}')
            if self.metric == 'F1_single':
                return score_list[0]
            else:
                return score


    def make_test_stage(self):
        dataloaders = self.get_dataloader('test', self.labels, self.CONFIG.fold, self.CONFIG.seed, self.test_path,
                                     self.CONFIG.batch_size)
        with torch.no_grad():
            pred_all = torch.Tensor()
            pred_all = pred_all.to(self.CONFIG.device)
            for i, (data, target) in tqdm(enumerate(dataloaders)):
                data = data.to(self.CONFIG.device)
                output = self.net(data)
                pred_all = torch.cat((pred_all, output), 0)

            output = torch.sigmoid(pred_all)

        pred = output.cpu().detach().numpy()
        return pred if self.CONFIG.probs else np.where(pred > 0.5, 1, 0)

    def _train_epoch(self):

        self.net.train()

        for i, (data, target) in tqdm(enumerate(self.train_loaders)):

            data = data.to(self.CONFIG.device)
            target = target.to(self.CONFIG.device)

            if self.CONFIG.MIX_UP and torch.rand(1)[0] < 0.2:
                mix_data, target_a, target_b, lam = mixup(data, target, alpha=0.5)
                output = self.net(mix_data)
                loss = self.criterion(output, target_a) * lam + (1 - lam) * self.criterion(output, target_b)
            else:
                output = self.net(data)
                loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _val_epoch(self):

        self.net.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                label_all = torch.Tensor().cuda()
                pred_all = torch.Tensor().cuda()
            else:
                label_all = torch.Tensor()
                pred_all = torch.Tensor()

            for i, (data, target) in tqdm(enumerate(self.val_loaders)):
                data = data.to(self.CONFIG.device)
                target = target.to(self.CONFIG.device)

                output = self.net(data)

                label_all = torch.cat((label_all, target), 0)
                pred_all = torch.cat((pred_all, output), 0)

            loss = self.criterion(pred_all, label_all)
            output = torch.sigmoid(pred_all)
            score = self.cal_score(label_all, output)

        if self.CONFIG.scheduler == 'ReduceLROnPlateau':
            self.scheduler.step(metrics=loss.item())
        else:
            self.scheduler.step()
        return score, loss.item()

    def _val_for_oof(self):
        self.net.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                label_all = torch.Tensor().cuda()
                pred_all = torch.Tensor().cuda()
            else:
                label_all = torch.Tensor()
                pred_all = torch.Tensor()

            for i, (data, target) in tqdm(enumerate(self.val_loaders)):
                data = data.to(self.CONFIG.device)
                target = target.to(self.CONFIG.device)

                output = self.net(data)

                label_all = torch.cat((label_all, target), 0)
                pred_all = torch.cat((pred_all, output), 0)

            output = torch.sigmoid(pred_all)

        return output.cpu().detach().numpy()

    def run(self):

        if self.CONFIG.one:
            self.best_score = [0] * 18
            for epoch in range(self.num_epochs):
                self._train_epoch()
                score, loss = self._val_epoch()
                self.LOGGER.info(f'epoch: {epoch}, loss: {loss}, score:{score}')
                for i in range(18):
                    if score[i] > self.best_score[i]:
                        self.LOGGER.info(f'score from {self.best_score} to {score} *')
                        self.best_score[i] = score[i]
                        torch.save(self.net.state_dict(), f"{self.CONFIG.model_dir}best_se_model_score_{self.fold}_{i}.pth")
                        self.LOGGER.info(f'fold {self.fold} best model saved!')

        else:
            for epoch in range(self.num_epochs):
                if self.early_stop_count > 8:
                    print('Early stop')
                    self.LOGGER.info(f'Early stop')
                    break
                else:
                    self._train_epoch()
                    score, loss = self._val_epoch()
                    self.LOGGER.info(f'epoch: {epoch}, loss: {loss}, score:{score}')
                    if self.metric == 'loss' and loss < self.best_loss:
                        self.best_loss = loss
                        torch.save(self.net.state_dict(), f"{self.CONFIG.model_dir}best_se_model_loss_{self.fold}.pth")
                    if score > self.best_score:
                        self.LOGGER.info(f'score from {self.best_score} to {score} *')
                        self.best_score = score
                        torch.save(self.net.state_dict(), f"{self.CONFIG.model_dir}best_se_model_score_{self.fold}.pth")
                        self.LOGGER.info(f'fold {self.fold} best model saved!')
                        self.early_stop_count = 0
                    else:
                        self.early_stop_count += 1

