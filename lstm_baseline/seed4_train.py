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
from torch.autograd import Variable

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

# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss, correct, total = 0, 0, 0
#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.squeeze().to(device)
#             preds = model(x)
#             loss = criterion(preds, y)
#             total_loss += loss.item()
#             _, predicted = torch.max(preds, 1)
#             correct += (predicted == y).sum().item()
#             total += y.size(0)
#     print("Test Total: " + str(total))
#     return total_loss / total, correct / total

def evaluate(model, loader, cuda):
    count = 0
    data_set_all = 0
    if cuda:
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        for _, (test_input, label) in enumerate(loader):
            if cuda:
                test_input, label = test_input.cuda(), label.cuda()
            test_input, label = Variable(test_input), Variable(label)
            data_set_all += len(label)
            x_shared_pred = model(test_input)
            _, pred = torch.max(x_shared_pred, dim=1)
            count += pred.eq(label.squeeze().data.view_as(pred)).sum()
    acc = float(count) / data_set_all
    print(f"Count: {count}, DataSet: {data_set_all}")
    return acc


def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_list = []
    for subject in range(15):  # SEED-IV 15 subjects
        print(f"=== Training, leaving subject {subject} for test ===")
        source_loaders, test_loader = getDataLoaders(subject, args)

        model = LSTMClassifier(input_size=310, hidden_size=128, num_layers=2, num_classes=4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss() # 这是什么？ 应该是分类问题的损失函数
        final_acc = 0.0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, source_loaders, optimizer, criterion, device)
            # test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            test_acc = evaluate(model, test_loader, True)
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
    parser.add_argument("--dataset_name", type=str, default="seed4")
    parser.add_argument("--path", type=str, default="/home/nise-emo/nise-lab/dataset/public/SEED_IV/eeg_feature_smooth/")
    parser.add_argument("--session", type=str, default="1")  
    parser.add_argument("--time_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers_train", type=int, default=4)
    parser.add_argument("--num_workers_test", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
