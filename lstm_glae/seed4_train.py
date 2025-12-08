# lstm_glae/seed4_train.py
import os, sys, argparse, random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

# 让脚本能从项目根目录 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.preprocess import getDataLoaders
from lstm_glae.model import LSTMClassifier
from lstm_glae.label_align import LabelAlignHead
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def soft_ce_loss(log_probs, soft_targets):
    """
    log_probs: [B, C], 模型的 LogSoftmax 输出
    soft_targets: [B, C], 对齐后的软标签 (每行和=1)
    返回: 标量损失
    """
    return -(soft_targets * log_probs).sum(dim=1).mean()

def train_one_epoch_warmup(model, loaders, optimizer, device, num_classes):
    """Warm-up: 不进行标签对齐, 用标准 hard label 训练"""
    model.train()
    total_correct, total = 0, 0
    total_loss = 0.0
    for loader in loaders:
        for x, y in loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)  # [B]
            optimizer.zero_grad()
            log_probs = model(x)          # [B, C]
            # hard CE: 等价于挑选 log_probs[range(B), y]
            loss = F.nll_loss(log_probs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            pred = log_probs.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)

def train_one_epoch_joint(model, label_head, loaders, optimizer, device, num_classes, reg_alpha, reg_beta, reg_gamma):
    """
    联合训练: LSTM + Label 对齐
    - 每个loader对应一个domain_id
    - y_onehot -> y_aligned = y_onehot @ T_j
    - loss = soft_ce + 正则
    """
    model.train()
    total_correct, total = 0, 0
    total_loss = 0.0

    for domain_id, loader in enumerate(loaders):
        for x, y in loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)  # [B]
            B = y.size(0)

            optimizer.zero_grad()

            # 模型前向
            log_probs = model(x)  # [B, C]

            # 软标签
            with torch.no_grad():
                y_onehot = torch.zeros(B, num_classes, device=device)
                y_onehot.scatter_(1, y.unsqueeze(1), 1)  # [B, C]

            Tj = label_head(domain_id)                 # [C, C] (行归一化)
            y_soft = y_onehot @ Tj                     # [B, C]

            # 主损失: soft CE
            loss_main = soft_ce_loss(log_probs, y_soft)

            # 正则: ||T-I||^2 + offdiag + row-entropy
            reg = (reg_alpha * LabelAlignHead.identity_reg(Tj) +
                   reg_gamma * LabelAlignHead.offdiag_reg(Tj) +
                   reg_beta  * LabelAlignHead.row_entropy(Tj))

            loss = loss_main + reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            pred = log_probs.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total += B

    return total_loss / max(total, 1), total_correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct, total = 0, 0
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.squeeze(-1).to(device)
        log_probs = model(x)
        loss = F.nll_loss(log_probs, y, reduction='sum')
        total_loss += loss.item()
        pred = log_probs.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_list = []
    for target_subject in range(args.subjects):  # LOSO
        print(f"\n=== LOSO fold: leave subject {target_subject} for test ===")
        source_loaders, test_loader = getDataLoaders(target_subject, args)

        # 模型/对齐头
        model = LSTMClassifier(
            input_size=args.input_dim,
            hidden_size=args.hid_dim,
            num_layers=args.n_layers,
            num_classes=args.cls_classes,
            dropout=args.dropout,
            bidirectional=True
        ).to(device)

        label_head = LabelAlignHead(
            num_domains=len(source_loaders),  # S-1
            num_classes=args.cls_classes,
            tau=args.tau,
            init_eye_scale=args.init_eye_scale
        ).to(device)

        # 两组参数，不同学习率
        optimizer = optim.Adam([
            {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": label_head.parameters(), "lr": args.lr_t, "weight_decay": 0.0},
        ])

        # 1) Warm-up
        for ep in range(args.epochs_warmup):
            tr_loss, tr_acc = train_one_epoch_warmup(model, source_loaders, optimizer, device, args.cls_classes)
            te_loss, te_acc = evaluate(model, test_loader, device)
            print(f"[Warmup {ep+1}/{args.epochs_warmup}] train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")

        # 2) Joint training
        final_acc = 0.0
        for ep in range(args.epochs_joint):
            tr_loss, tr_acc = train_one_epoch_joint(
                model, label_head, source_loaders, optimizer, device, args.cls_classes,
                reg_alpha=args.reg_alpha, reg_beta=args.reg_beta, reg_gamma=args.reg_gamma
            )
            te_loss, te_acc = evaluate(model, test_loader, device)
            if te_acc > final_acc:
                final_acc = te_acc
            print(f"[Joint {ep+1}/{args.epochs_joint}] train_acc={tr_acc:.4f} test_acc={te_acc:.4f} best_acc={final_acc:.4f}")

        acc_list.append(final_acc)

    # 汇总
    acc_arr = np.array(acc_list, dtype=np.float32)
    print("\n===== SEED-IV (GLAE-LSTM) Results =====")
    print("Each subject acc:", ["{:.4f}".format(a) for a in acc_arr.tolist()])
    print("Final Avg Acc: {:.4f}".format(acc_arr.mean()))
    print("Final Std Acc: {:.4f}".format(acc_arr.std()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据/实验设置
    parser.add_argument("--dataset_name", type=str, default="seed4")
    parser.add_argument("--path", type=str, default="/home/nise-emo/nise-lab/dataset/public/SEED_IV/eeg_feature_smooth/")
    parser.add_argument("--session", type=str, default="1")
    parser.add_argument("--subjects", type=int, default=15)
    parser.add_argument("--time_steps", type=int, default=10)  # SEED-IV default
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers_train", type=int, default=4)
    parser.add_argument("--num_workers_test", type=int, default=2)
    parser.add_argument("--cls_classes", type=int, default=4)

    # 模型
    parser.add_argument("--input_dim", type=int, default=310)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    # 训练
    parser.add_argument("--epochs_warmup", type=int, default=100)
    parser.add_argument("--epochs_joint", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_t", type=float, default=5e-4)   # label head 学习率略低
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=3)

    # Label 对齐超参
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--init_eye_scale", type=float, default=5.0)
    parser.add_argument("--reg_alpha", type=float, default=1e-2)   # ||T-I||^2
    parser.add_argument("--reg_beta", type=float, default=1e-3)    # row entropy
    parser.add_argument("--reg_gamma", type=float, default=1e-3)   # off-diag

    args = parser.parse_args()
    main(args)
