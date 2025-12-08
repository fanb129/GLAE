import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import defaultdict


def train_one_epoch(model, loaders, optimizer, device, epoch, epochs):
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

