import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from toolbox import get_dataset, load_ckpt
from toolbox.optim.Ranger import Ranger
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox.utils import save_ckpt
from toolbox import setup_seed
from toolbox.datasets.irseg import IRSeg
import matplotlib.pyplot as plt
from PIL import Image

from net import SGFNet

import os
import warnings

warnings.filterwarnings('ignore')

######################## 超参数设置 #########################
setup_seed(99)
batch_size = 2
lr_start = 4e-4
weight_decay = 0.05
epoch = 90
gpu = '0'
saveprefix = ''
loadmodel =  ''

def calculate_mae(pred, target):
    prob = torch.sigmoid(pred)
    return torch.mean(torch.abs(prob - target)).item()


def calculate_fmeasure(pred, target, beta=0.3, threshold=0.5):
    prob = torch.sigmoid(pred)
    binary_pred = (prob > threshold).float()
    target_bin = (target > 0.5).float() 

    TP = (binary_pred * target_bin).sum(dim=(1, 2, 3))
    FP = (binary_pred * (1 - target_bin)).sum(dim=(1, 2, 3))
    FN = ((1 - binary_pred) * target_bin).sum(dim=(1, 2, 3))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    f_beta = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-6)
    return f_beta.mean().item()


def calculate_smeasure(pred, target, alpha=0.5):
    def _s_object(pred, gt):
        if gt.numel() == 0 or pred.numel() == 0:
            return torch.tensor(0.0).to(pred.device)

        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = _object(fg, gt)
        o_bg = _object(bg, 1 - gt)
        u = gt.mean()
        return u * o_fg + (1 - u) * o_bg

    def _object(pred, gt):
        temp = pred[gt == 1]
        if temp.numel() == 0:
            return torch.tensor(0.0).to(pred.device)
        x = temp.mean()
        sigma_x = temp.std()
        return 2.0 * x / (x ** 2 + 1.0 + sigma_x + 1e-20)

    def _s_region(pred, gt):
        X, Y = _centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
        p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
        q1 = _ssim(p1, gt1)
        q2 = _ssim(p2, gt2)
        q3 = _ssim(p3, gt3)
        q4 = _ssim(p4, gt4)
        return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    def _centroid(gt):
        rows, cols = gt.shape[-2:]
        if gt.sum() == 0:
            return cols // 2, rows // 2
        else:
            total = gt.sum()
            i = torch.arange(cols, device=gt.device).float()
            j = torch.arange(rows, device=gt.device).float()
            X = (gt.sum(dim=-2) * i).sum() / total
            Y = (gt.sum(dim=-1) * j).sum() / total
            return int(X.round().item()), int(Y.round().item())

    def _divideGT(gt, X, Y):
        h, w = gt.shape[-2:]
        area = h * w
        LT = gt[..., :Y, :X]
        RT = gt[..., :Y, X:]
        LB = gt[..., Y:, :X]
        RB = gt[..., Y:, X:]
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(pred, X, Y):
        return pred[..., :Y, :X], pred[..., :Y, X:], pred[..., Y:, :X], pred[..., Y:, X:]

    def _ssim(pred, gt):
        if pred.numel() == 0 or gt.numel() == 0:
            return torch.tensor(0.0).to(pred.device)

        mu_x = pred.mean()
        mu_y = gt.mean()
        sigma_x = pred.std()
        sigma_y = gt.std()
        sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()

        denominator = (mu_x ** 2 + mu_y ** 2 + 1e-6) * (sigma_x ** 2 + sigma_y ** 2 + 1e-6)
        return (4 * sigma_xy * mu_x * mu_y) / denominator

    device = pred.device
    prob = torch.sigmoid(pred).to(device)
    target_bin = (target > 0.5).float()

    batch_size = prob.shape[0]
    total_S = torch.tensor(0.0, device=device)
    for i in range(batch_size):

        pred_i = prob[i].squeeze().to(device)
        gt_i = target_bin[i].squeeze().to(device)

        if gt_i.sum() == 0 or gt_i.sum() == gt_i.numel():
            total_S += torch.tensor(1.0, device=device)
        else:
            Q = alpha * _s_region(pred_i, gt_i) + (1 - alpha) * _s_object(pred_i, gt_i)
            total_S += Q.clamp(min=0)


    return (total_S / batch_size).item()


def calculate_emeasure(pred, target):

    def _eval_e(pred, gt, num=255):
        device = pred.device
        pred = pred.to(device)
        gt = gt.to(device)

        if gt.numel() == 0 or pred.numel() == 0:
            return torch.tensor(0.0).to(device)

        score = torch.zeros(num, device=device)
        thlist = torch.linspace(0, 1 - 1e-10, num, device=device)

        gt_mean = gt.mean() if gt.numel() > 0 else 0.0
        for i in range(num):
            pred_th = (pred > thlist[i]).float()
            if gt_mean < 1e-6:
                enhanced = 1 - pred_th
            elif gt_mean > 1 - 1e-6:
                enhanced = pred_th
            else:
                fm = pred_th - pred_th.mean()
                gt = gt - gt.mean()
                align = 2 * gt * fm / (gt  ** 2 + fm  ** 2 + 1e-20)
                enhanced = ((align + 1)  ** 2) / 4
            score[i] = enhanced.mean()
        return score

    device = pred.device
    prob = torch.sigmoid(pred).to(device)
    target_bin = (target > 0.5).float().to(device)

    batch_size = prob.shape[0]
    total_E = 0.0
    for i in range(batch_size):
        pred_i = prob[i].to(device)
        gt_i = target_bin[i].to(device)

        if gt_i.numel() == 0 or pred_i.numel() == 0:
            continue

        E = _eval_e(pred_i, gt_i)
        total_E += E.max().clamp(min=0, max=1)

    return (total_E / batch_size).item() if batch_size > 0 else 0.0


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

class eeemodelLoss(nn.Module):
    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()
        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.CE = nn.BCEWithLogitsLoss()
        self.ECE = nn.BCELoss()

    def forward(self, predict, targets):
        loss = self.CE(predict, targets) + iou_loss(predict, targets)
        return loss


def preprocess_targets(label, threshold=127):
    targets = label.unsqueeze(1).float()
    return (targets > threshold).float()


def run():
    model = SGFNet(1).cuda()

    if loadmodel != '':
        load_ckpt(model=model, prefix=loadmodel)
    
    trainset = IRSeg(mode='train')
    testset = IRSeg(mode='test')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                             drop_last=True)

    optimizer = Ranger(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / epoch) ** 0.9)

    metrics = {
        'train': {k: averageMeter() for k in ['loss', 'mae', 'fmeasure', 'smeasure', 'emeasure']},
        'test': {k: averageMeter() for k in ['loss', 'mae', 'fmeasure', 'smeasure', 'emeasure']}
    }
    best_metrics = {'mae': float('inf'), 'fmeasure': 0.0, 'smeasure': 0.0, 'emeasure': 0.0}
    running_metrics_test = runningScore(2, ignore_index=-1)

    for ep in range(epoch):
        model.train()
        [m.reset() for m in metrics['train'].values()]

        for i, sample in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            image = sample['image'].cuda()
            DOP = sample['DOP'].cuda().repeat(1, 3, 1, 1)
            label = sample['label'].cuda()
            targets = preprocess_targets(label)

            predict = model(image, DOP)[0]
            predict = predict[:, 0:1, :, :]

            loss = eeemodelLoss()(predict, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metrics['train']['loss'].update(loss.item())
                metrics['train']['mae'].update(calculate_mae(predict, targets))
                metrics['train']['fmeasure'].update(calculate_fmeasure(predict, targets))
                metrics['train']['smeasure'].update(calculate_smeasure(predict, targets))
                metrics['train']['emeasure'].update(calculate_emeasure(predict, targets))

        model.eval()
        [m.reset() for m in metrics['test'].values()]

        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                image = sample['image'].cuda()
                DOP = sample['DOP'].cuda().repeat(1, 3, 1, 1)
                label = sample['label'].cuda()
                targets = preprocess_targets(label)

                predict = model(image, DOP)[0]
                predict = predict[:, 0:1, :, :]

                loss = eeemodelLoss()(predict, targets)
                metrics['test']['loss'].update(loss.item())
                metrics['test']['mae'].update(calculate_mae(predict, targets))
                metrics['test']['fmeasure'].update(calculate_fmeasure(predict, targets))
                metrics['test']['smeasure'].update(calculate_smeasure(predict, targets))
                metrics['test']['emeasure'].update(calculate_emeasure(predict, targets))

                predicted = (torch.sigmoid(predict) > 0.5).float()
                running_metrics_test.update(targets.cpu().numpy().astype(int),
                                            predicted.cpu().numpy().astype(int))

        test_metrics = running_metrics_test.get_scores()[0]
        test_miou = test_metrics["IoU: "]
        test_macc = test_metrics["class_acc: "]

        current_metrics = {
            'mae': metrics['test']['mae'].avg,
            'fmeasure': metrics['test']['fmeasure'].avg,
            'smeasure': metrics['test']['smeasure'].avg,
            'emeasure': metrics['test']['emeasure'].avg
        }
        if ep >= 50: save_ckpt(model,
                              prefix=f"model//epoch_{ep + 1}_mae_{current_metrics['mae']:.4f}_f_{current_metrics['fmeasure']:.4f}_e_{current_metrics['emeasure']:.4f}_s_{current_metrics['smeasure']:.4f}"
                              )

        log_msg = f'Epoch [{ep + 1:03d}/{epoch}] | '
        log_msg += f'Train: L={metrics["train"]["loss"].avg:.3f} MAE={metrics["train"]["mae"].avg:.3f} '
        log_msg += f'F={metrics["train"]["fmeasure"].avg:.3f} S={metrics["train"]["smeasure"].avg:.3f} E={metrics["train"]["emeasure"].avg:.3f} | '
        log_msg += f'Test: L={metrics["test"]["loss"].avg:.3f} MAE={current_metrics["mae"]:.3f} '
        log_msg += f'F={current_metrics["fmeasure"]:.3f} S={current_metrics["smeasure"]:.3f} E={current_metrics["emeasure"]:.3f} | '
        log_msg += f'mIoU: {test_miou:.3f} mAcc: {test_macc:.3f}'
        print(log_msg)

        scheduler.step()


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    run()

