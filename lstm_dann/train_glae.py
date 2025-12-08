import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import defaultdict


def train_all_one_epoch(model, loaders, optimizer, device, epoch, epochs):
    iteration = len(loaders[0]) + 1
    source_iters = []
    for i in range(len(loaders)):
        source_iters.append(iter(loaders[i]))
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i in range(1, iteration + 1):
        p = float(i + epoch * iteration) / epochs / iteration
        m = 2. / (1. + np.exp(-10 * p)) - 1
        for j in range(len(source_iters)):
            try:
                x, y = next(source_iters[j])
            except:
                source_iters[j] = iter(loaders[j])
                x, y = next(source_iters[j])
            x, y = x.to(device), y.squeeze().to(device)
            subject_id = ((torch.ones(y.size(0)) * j).long()).to(device)
            optimizer.zero_grad()
            emotion_output, domain_output, features = model(x, m)
            dann_loss = F.nll_loss(domain_output, subject_id)
            cls_loss = F.nll_loss(emotion_output, y)
            loss = cls_loss + 0.05 * dann_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train_cls_one_epoch(model, loaders, optimizer, device, epoch, epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for loader in loaders:
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            optimizer.zero_grad()
            emotion_output, _, _ = model(x, 0)
            loss = F.nll_loss(emotion_output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train_label_align_one_epoch(model, optimizer, target_domain, predictions, true_labels, label_align_loss_fn):
    model.train()
    optimizer.zero_grad()
    # 获取变换矩阵
    T_target = model.get_T(target_domain)
    
    # 应用变换
    aligned_labels = torch.matmul(true_labels, T_target)

    # 添加epsilon防止数值问题
    predictions = predictions + 1e-6  # 增大epsilon
    predictions = predictions / predictions.sum(dim=1, keepdim=True)
    
    # 计算损失（需要梯度）
    loss = label_align_loss_fn(aligned_labels, predictions, [T_target])
    loss.backward()
    optimizer.step()
    return loss


def train_cls_one_epoch_with_aligned_label(model, loaders, optimizer, device, label_align_model, anchor_id):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for domain_id, loader in enumerate(loaders):
        if domain_id == anchor_id:
            continue
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            # 获取对齐后的标签
            with torch.no_grad():
                aligned_labels = label_align_model.get_aligned_label(y, domain_id)
            optimizer.zero_grad()
            emotion_output, _, _ = model(x, 0)
            # KL散度
            loss = -(aligned_labels * emotion_output).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train_all_one_epoch_with_aligned_label(model, loaders, optimizer, device, epoch, epochs, label_align_model):
    iteration = len(loaders[0]) + 1
    source_iters = []
    for i in range(len(loaders)):
        source_iters.append(iter(loaders[i]))
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i in range(1, iteration + 1):
        p = float(i + epoch * iteration) / epochs / iteration
        m = 2. / (1. + np.exp(-10 * p)) - 1
        for j in range(len(source_iters)):
            try:
                x, y = next(source_iters[j])
            except:
                source_iters[j] = iter(loaders[j])
                x, y = next(source_iters[j])
            x, y = x.to(device), y.squeeze().to(device)
            subject_id = ((torch.ones(y.size(0)) * j).long()).to(device)

            # 获取对齐后的标签
            with torch.no_grad():
                aligned_labels = label_align_model.get_aligned_label(y, j)

            optimizer.zero_grad()
            emotion_output, domain_output, features = model(x, m)
            dann_loss = F.nll_loss(domain_output, subject_id)
            cls_loss = -(aligned_labels * emotion_output).sum(dim=1).mean()
            loss = cls_loss + 0.05 * dann_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            emotion_output, _, _ = model(x)
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

