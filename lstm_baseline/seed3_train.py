# lstm_baseline/seed3_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.preprocess import getDataLoaders
from model import LSTMClassifier
import argparse
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loaders, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for loader in loaders:
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_list = []
    for subject in range(15):  # LOSO: leave-one-subject-out
        print(f"=== Training, leaving subject {subject} for test ===")
        source_loaders, test_loader = getDataLoaders(subject, args)

        model = LSTMClassifier(input_size=310, hidden_size=128, num_layers=2, num_classes=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()
        final_acc = 0.0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, source_loaders, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            if test_acc > final_acc:
                final_acc = test_acc
            print(f"[Epoch {epoch+1}/{args.epochs}] Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} Best Acc={final_acc:.4f}")

        acc_list.append(final_acc)
    
    print("final acc avg:", str(np.average(acc_list)))
    print("final acc std:", str(np.std(acc_list)))
    print('final each acc:')
    # 打印索引和值
    for i,acc in enumerate(acc_list):
        print(f'Subject {i+1}: {acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="seed3")
    parser.add_argument("--path", type=str, default="/home/fb/src/dataset/SEED/ExtractedFeatures/")
    parser.add_argument("--session", type=str, default="1")  # 只跑Session1
    parser.add_argument("--time_steps", type=int, default=30)  # 窗口长度
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_test", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
