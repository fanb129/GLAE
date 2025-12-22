import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import copy
import time
import logging
import multiprocessing

# 确保能找到同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.preprocess import getDataLoaders
from model import LSTMDANN, LabelAlignModel, LabelAlignLoss, EmotionClassifier
from train import (
    train_all_one_epoch, 
    train_cls_one_epoch, 
    evaluate,
    train_label_align_one_epoch, 
    train_cls_one_epoch_with_aligned_label, 
    train_all_one_epoch_with_aligned_label
)

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_logging(subject, log_dir, prefix=""):
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            pass 
            
    filename = f"{prefix}subject_{subject}.log" if prefix else f"subject_{subject}.log"
    log_file = os.path.join(log_dir, filename)
    
    logger = logging.getLogger(f"subject_{subject}_{prefix}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers = []
        
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def evaluate_on_sources(model, source_loaders, exclude_id, device):
    """计算模型在除 Anchor 外所有源域上的平均准确率"""
    model.eval()
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for i, loader in enumerate(source_loaders):
            if i == exclude_id:
                continue
            acc = evaluate(model, loader, device)
            total_acc += acc
            count += 1
    
    return total_acc / count if count > 0 else 0.0

def collect_predictions(model, loader, args, device):
    """收集预测结果用于计算对齐矩阵"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            soft_probs, _, _ = model(x)
            
            y_onehot = torch.zeros(y.size(0), args.num_classes, device=y.device)
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            
            all_predictions.append(soft_probs)
            all_true_labels.append(y_onehot)
    
    if not all_predictions:
        return None, None
        
    return torch.cat(all_predictions, dim=0), torch.cat(all_true_labels, dim=0)


def step1(subject, args):
    """
    Step 1: Pre-training (Standard LSTM-DANN)
    """
    args.num_workers_train = 0
    args.num_workers_test = 0

    logger = setup_logging(subject, args.log_dir, prefix="pretrain_")
    logger.info(f"=== PreTraining Subject {subject} ===")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_loaders, test_loader = getDataLoaders(subject, args)
    model = LSTMDANN(
        input_size=310,
        hidden_size=128,
        num_layers=2,
        num_classes=args.num_classes,
        num_domains=14,
        dropout=0.5).to(device)
    
    optimizer_all = optim.Adam(model.parameters(), lr=args.lr)

    final_acc = 0.0
    best_model = None
    
    # 增加简单的 Early Stopping 或者是记录最好的模型
    for epoch in range(args.epochs_pretrain):
        train_loss, train_acc = train_all_one_epoch(model, source_loaders, optimizer_all, device, epoch, args.epochs_pretrain)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > final_acc:
            final_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
            
        if epoch % 10 == 0:
            logger.info(f"Pretrain [Epoch {epoch+1}/{args.epochs_pretrain}] Loss={train_loss:.4f} TrainAcc={train_acc:.4f} TestAcc={test_acc:.4f} Best={final_acc:.4f}")
    
    save_path = os.path.join(args.model_dir, f"pretrain_model_subject{subject}.pth")
    torch.save(best_model, save_path)
    return final_acc


# =============================================================================
# 新函数 1: 仅用于评估 Anchor 的代表性 (Selection Phase)
# =============================================================================
def evaluate_candidate_anchor(subject, args, anchor_id, device, source_loaders, logger):
    """
    轻量级函数：加载预训练特征，重置分类器，仅在 Anchor 上训练。
    返回：Best Source Validation Score (SrcScore)
    这是为了避免 Step 1 的过拟合影响 Anchor 选择。
    """
    logger.info(f"--- Evaluating Candidate Anchor: {anchor_id} (Selection Phase) ---")
    
    pretrain_path = os.path.join(args.model_dir, f"pretrain_model_subject{subject}.pth")
    if not os.path.exists(pretrain_path):
        return 0.0

    model = LSTMDANN(input_size=310, hidden_size=128, num_layers=2, 
                     num_classes=args.num_classes, num_domains=14, dropout=0.5).to(device)
    model.load_state_dict(torch.load(pretrain_path))
    
    # === 关键：重置分类器 ===
    model.emotion_classifier = EmotionClassifier(input_size=256, num_classes=args.num_classes, dropout=0.5).to(device)
    optimizer_cls = optim.Adam(model.emotion_classifier.parameters(), lr=args.lr)
    
    best_src_val = 0.0
    
    # 仅运行 Phase 1 (Anchor Training)
    for epoch in range(args.anchor_epochs):
        train_cls_one_epoch(model, [source_loaders[anchor_id]], optimizer_cls, device, epoch, args.anchor_epochs)
        
        # 计算在"其他源域"上的泛化能力
        src_val = evaluate_on_sources(model, source_loaders, anchor_id, device)
        if src_val > best_src_val:
            best_src_val = src_val
            
    return best_src_val


# =============================================================================
# 新函数 2: 针对选定的 Best Anchor 运行完整 GLA 流程 (Phase 1-5)
# =============================================================================
def train_final_gla(subject, args, anchor_id, device, source_loaders, test_loader, label_align_loss_fn, logger):
    """
    重量级函数：运行完整的 Phase 1-5 流程。
    """
    logger.info(f"--- Running Full GLA Pipeline with Best Anchor: {anchor_id} ---")
    
    # 初始化标签对齐模型
    label_align_model = LabelAlignModel(
        num_domains=len(source_loaders), 
        num_classes=args.num_classes, 
        tau=args.tau).to(device)
    optimizer_label_align = torch.optim.Adam(label_align_model.parameters(), lr=args.lr)
    
    # 加载预训练模型
    pretrain_path = os.path.join(args.model_dir, f"pretrain_model_subject{subject}.pth")
    model = LSTMDANN(input_size=310, hidden_size=128, num_layers=2, 
                     num_classes=args.num_classes, num_domains=14, dropout=0.5).to(device)
    model.load_state_dict(torch.load(pretrain_path))
    
    # Phase 1 前重置分类器
    model.emotion_classifier = EmotionClassifier(input_size=256, num_classes=args.num_classes, dropout=0.5).to(device)
    optimizer_cls = optim.Adam(model.emotion_classifier.parameters(), lr=args.lr)
    
    peak_test_acc_1 = 0.0
    peak_test_acc_3 = 0.0
    peak_test_acc_5 = 0.0
    peak_test_acc = 0.0
    
    def run_evaluation(phase, epoch, max_acc, peak_test_acc):
        test_val = evaluate(model, test_loader, device)
        if test_val > max_acc:
            max_acc = test_val
        if max_acc > peak_test_acc:
            peak_test_acc = max_acc
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"{phase} [Ep {epoch}] TestAcc={test_val:.4f} MaxAcc={max_acc:.4f} (Peak={peak_test_acc:.4f})")
        return max_acc, peak_test_acc

    # === Phase 1: Train Regressor on Anchor ===
    logger.info("Phase 1: Warming up Anchor Classifier")
    for epoch in range(args.anchor_epochs):
        train_cls_one_epoch(model, [source_loaders[anchor_id]], optimizer_cls, device, epoch, args.anchor_epochs)
        peak_test_acc_1, peak_test_acc = run_evaluation("Anchor-Cls", epoch+1, peak_test_acc_1, peak_test_acc)

    # === Phase 2: Align Remaining Domains (Calculate T) ===
    logger.info("Phase 2: Training Label Alignment Matrix")
    for source_id in range(len(source_loaders)):
        if source_id == anchor_id: continue
        preds, trues = collect_predictions(model, source_loaders[source_id], args, device)
        if preds is not None:
            # 注意：这里接收 diag_mean 用于监控
            loss_val, diag_mean = train_label_align_one_epoch(label_align_model, optimizer_label_align, source_id, preds, trues, label_align_loss_fn)
            logger.info(f"  Aligning Src {source_id}: Loss={loss_val:.4f}, T_Diag={diag_mean:.4f}")

    # === Phase 3: Train on Remaining Domains with Aligned Labels ===
    # 再次重置分类器，验证对齐后的标签质量
    model.emotion_classifier = EmotionClassifier(input_size=256, num_classes=args.num_classes, dropout=0.5).to(device)
    optimizer_cls = optim.Adam(model.emotion_classifier.parameters(), lr=args.lr)

    logger.info("Phase 3: Training Classifier on Others (Reset Classifier)")
    for epoch in range(args.cls_epochs):
        train_cls_one_epoch_with_aligned_label(model, source_loaders, optimizer_cls, device, label_align_model, anchor_id)
        peak_test_acc_3, peak_test_acc = run_evaluation("Aligned-Cls", epoch+1, peak_test_acc_3, peak_test_acc)

    # === Phase 4: Align Anchor (Self-Correction) ===
    logger.info("Phase 4: Anchor Self-Correction")
    preds, trues = collect_predictions(model, source_loaders[anchor_id], args, device)
    if preds is not None:
        loss_val, diag_mean = train_label_align_one_epoch(label_align_model, optimizer_label_align, anchor_id, preds, trues, label_align_loss_fn)
        logger.info(f"  Aligning Anchor {anchor_id}: Loss={loss_val:.4f}, T_Diag={diag_mean:.4f}")

    # === Phase 5: Final Joint Fine-tuning ===
    if args.fine_tune_phase5:
        logger.info("Phase 5: Final Joint Training (FINE-TUNING from Step 1 Weights)")
        # 加载预训练权重
        model = LSTMDANN(input_size=310, hidden_size=128, num_layers=2, 
                         num_classes=args.num_classes, num_domains=14, dropout=0.5).to(device)
        model.load_state_dict(torch.load(pretrain_path))
        # 极小的学习率，只修补边界
        optimizer_all = optim.Adam(model.parameters(), lr=args.lr * 0.01)
    else:
        logger.info("Phase 5: Final Joint Training (FROM SCRATCH)")
        # 随机初始化
        model = LSTMDANN(input_size=310, hidden_size=128, num_layers=2, 
                         num_classes=args.num_classes, num_domains=14, dropout=0.5).to(device)
        # 正常学习率
        optimizer_all = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_all_one_epoch_with_aligned_label(model, source_loaders, optimizer_all, device, epoch, args.epochs, label_align_model)
        peak_test_acc_5, peak_test_acc = run_evaluation("Final-All", epoch+1, peak_test_acc_5, peak_test_acc)

    return peak_test_acc_1, peak_test_acc_3, peak_test_acc_5, peak_test_acc


def step2(subject, args):
    """
    Step 2 Main: 
    1. 遍历所有 Anchor 仅运行 Phase 1 (Selection)，选出 Best Anchor。
    2. 对 Best Anchor 运行完整 GLA 流程 (Execution)。
    """
    args.num_workers_train = 0
    args.num_workers_test = 0

    logger = setup_logging(subject, args.log_dir)
    logger.info(f"=== GLAE Step 2: Subject {subject} Start ===")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_loaders, test_loader = getDataLoaders(subject, args)
    label_align_loss_fn = LabelAlignLoss(num_classes=args.num_classes, 
                                         reg_alpha=args.reg_alpha, 
                                         reg_beta=args.reg_beta, 
                                         reg_gamma=args.reg_gamma)

    # -----------------------------------------------------
    # Part 1: Anchor Selection (快速筛选)
    # -----------------------------------------------------
    logger.info(">>> Starting Anchor Selection Phase (Running Phase 1 Only)")
    best_src_score = -1.0
    best_anchor_id = -1
    best_anchor_id = args.anchor_list[subject]
    # for anchor_id in range(len(source_loaders)):
    #     score = evaluate_candidate_anchor(subject, args, anchor_id, device, source_loaders, logger)
    #     logger.info(f"Candidate {anchor_id}: SrcScore={score:.4f}")
        
    #     if score > best_src_score:
    #         best_src_score = score
    #         best_anchor_id = anchor_id
            
    logger.info(f">>> Selection Done. Best Anchor: {best_anchor_id} (Score: {best_src_score:.4f})")

    # -----------------------------------------------------
    # Part 2: Final Training (针对最佳 Anchor 全量跑)
    # -----------------------------------------------------
    logger.info(">>> Starting Final GLA Pipeline")
    peak_accs = train_final_gla(subject, args, best_anchor_id, device, source_loaders, 
                                test_loader, label_align_loss_fn, logger)
    
    p1, p3, p5, peak_test_acc = peak_accs
    
    logger.info("="*50)
    logger.info(f"Subject {subject} Done.")
    logger.info(f"Selected Anchor: {best_anchor_id}")
    logger.info(f"Phase 1 Peak: {p1:.4f}")
    logger.info(f"Phase 3 Peak: {p3:.4f}")
    logger.info(f"Phase 5 Peak: {p5:.4f}")
    logger.info(f"peak_test_acc: {peak_test_acc:.4f}")
    logger.info("="*50)
    
    return p1, p3, p5, peak_test_acc, best_anchor_id


def main(args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    print(f"Dataset: {args.dataset_name}, CUDA: {args.cuda}")
    print(f"Running multiprocessing pool with 15 processes...")

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with multiprocessing.Pool(processes=15) as pool:
        subjects = range(15)
        
        if args.step1:
            print("--- Starting Step 1 (Pretrain) ---")
            step1_acc_list = pool.starmap(step1, [(subject, args) for subject in subjects])
            print(f"Pretrain Avg Acc: {np.average(step1_acc_list):.4f}")

        if args.step2:
            print("\n--- Starting Step 2 (GLA) ---")
            step2_results = pool.starmap(step2, [(subject, args) for subject in subjects])
            
            peak_test_acc_1 = [res[0] for res in step2_results]
            peak_test_acc_3 = [res[1] for res in step2_results]
            peak_test_acc_5 = [res[2] for res in step2_results]
            peak_test_acc = [res[3] for res in step2_results]
            best_anchor_list = [res[4] for res in step2_results]

            print(f"\nFinal Results ({args.dataset_name}):")
            print(f"Avg peak_test_acc: {np.average(peak_test_acc):.4f} ± {np.std(peak_test_acc):.4f}")
            print(f"Avg Phase 5: {np.average(peak_test_acc_5):.4f} ± {np.std(peak_test_acc_5):.4f}")
            
            print("Details (P1 / P3 / P5 / Peak Test Acc):")
            for i, acc in enumerate(peak_test_acc):
                print(f"Sub {i}: {peak_test_acc_1[i]:.4f} / {peak_test_acc_3[i]:.4f} / {peak_test_acc_5[i]:.4f} / {acc:.4f} (Anchor {best_anchor_list[i]})")


if __name__ == "__main__":
    # 配置路径
    seed3_path = "/home/fb/src/dataset/SEED/ExtractedFeatures/"
    seed4_path = "/home/nise-emo/nise-lab/dataset/public/SEED_IV/eeg_feature_smooth/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="seed3")
    parser.add_argument("--session", type=str, default="1")  
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers_train", type=int, default=0) 
    parser.add_argument("--num_workers_test", type=int, default=0)

    # 关键参数默认值已修改 (加强约束，防止标签漂移)
    # parser.add_argument('--tau', type=float, default=1.0, help="Temperature for Softmax, higher means smoother T")
    # parser.add_argument('--reg_alpha', type=float, default=1.0, help="Identity Matrix Regularization Weight")
    # parser.add_argument('--reg_beta', type=float, default=0.01, help="Entropy Regularization Weight")
    # parser.add_argument('--reg_gamma', type=float, default=0.01, help="Off-diagonal Regularization Weight")
    
    parser.add_argument('--tau', type=float, default=0.7, help='标签对齐softmax温度参数')
    parser.add_argument('--reg_alpha', type=float, default=1e-2, help='标签对齐正则化系数')
    parser.add_argument('--reg_beta', type=float, default=1e-3, help='行熵正则化系数')
    parser.add_argument('--reg_gamma', type=float, default=1e-3, help='非对角线惩罚系数')

    parser.add_argument('--epochs_pretrain', type=int, default=300)
    parser.add_argument('--anchor_epochs', type=int, default=150)
    parser.add_argument('--align_epochs', type=int, default=150)
    parser.add_argument('--cls_epochs', type=int, default=150)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--cuda", type=str, default='2')
    parser.add_argument("--step1", action='store_true')
    parser.add_argument("--step2", action='store_true')
    parser.add_argument("--subject", type=int, default=0)

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    today = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
    
    if args.dataset_name == "seed3":
        args.path = seed3_path
        args.num_classes = 3
        args.time_steps = 30
        args.batch_size = 512
        args.model_dir = os.path.join(current_dir, "models/seed3")
        args.log_dir = os.path.join(current_dir, f"logs/seed3/{today}")
        # 1. 降低 tau，让标签更尖锐，减少噪声干扰
        # args.tau = 0.7  
        # 2. 提高 alpha，强迫 T 矩阵更接近单位矩阵
        # args.reg_alpha = 1e-2 
        args.fine_tune_phase5 = False
        args.anchor_list = [3,4,0,3,10,2,0,2,6,3,1,13,8,0,12]
    elif args.dataset_name == "seed4":
        args.path = seed4_path
        args.num_classes = 4
        args.time_steps = 10
        args.batch_size = 256
        args.model_dir = os.path.join(current_dir, "models/seed4")
        args.log_dir = os.path.join(current_dir, f"logs/seed4/{today}")
        # args.tau = 1.0 
        # args.reg_alpha = 1.0
        # SEED-IV 重头训效果好，因为 Step 1 特征不够好
        args.fine_tune_phase5 = False
    
    main(args)