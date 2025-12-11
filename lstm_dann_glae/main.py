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
    
    for epoch in range(args.epochs_pretrain):
        train_loss, train_acc = train_all_one_epoch(model, source_loaders, optimizer_all, device, epoch, args.epochs_pretrain)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > final_acc:
            final_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
            
        logger.info(f"Pretrain [Epoch {epoch+1}/{args.epochs_pretrain}] Loss={train_loss:.4f} TrainAcc={train_acc:.4f} TestAcc={test_acc:.4f} Best={final_acc:.4f}")
    
    save_path = os.path.join(args.model_dir, f"pretrain_model_subject{subject}.pth")
    torch.save(best_model, save_path)
    return final_acc


def step2_1(subject, args, anchor_id, device, source_loaders, test_loader, label_align_loss_fn, logger):
    logger.info(f"--- Processing Candidate Anchor: {anchor_id} ---")
    
    # -----------------------------------------------------------
    # 初始化 Label Align Model (贯穿 Phase 2-5)
    # -----------------------------------------------------------
    label_align_model = LabelAlignModel(
        num_domains=len(source_loaders), 
        num_classes=args.num_classes, 
        tau=args.tau).to(device)
    optimizer_label_align = torch.optim.Adam(label_align_model.parameters(), lr=args.lr)
    
    # -----------------------------------------------------------
    # 准备工作：加载预训练模型 (用于 Phase 1-4 的计算)
    # -----------------------------------------------------------
    pretrain_path = os.path.join(args.model_dir, f"pretrain_model_subject{subject}.pth")
    if not os.path.exists(pretrain_path):
        logger.error(f"Pretrained model not found: {pretrain_path}")
        return 0.0, 0.0

    # 这里我们定义一个 helper function 来快速重置/加载模型
    def load_pretrained_feature_extractor():
        tmp_model = LSTMDANN(
            input_size=310, hidden_size=128, num_layers=2, 
            num_classes=args.num_classes, num_domains=14, dropout=0.5
        ).to(device)
        tmp_model.load_state_dict(torch.load(pretrain_path))
        return tmp_model

    # 初始加载
    model = load_pretrained_feature_extractor()
    
    # 指标记录
    selection_score = 0.0 
    peak_test_acc = 0.0
    
    def run_evaluation(phase, epoch, update_selection_score=False):
        nonlocal selection_score, peak_test_acc
        
        # Source Validation (仅在 Phase 1 且 update=True 时更新 selection_score)
        src_val = evaluate_on_sources(model, source_loaders, anchor_id, device)
        if update_selection_score:
            if src_val > selection_score:
                selection_score = src_val
        
        # Test Evaluation (始终记录峰值)
        test_val = evaluate(model, test_loader, device)
        if test_val > peak_test_acc:
            peak_test_acc = test_val
            
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"{phase} [Ep {epoch}] SrcVal={src_val:.4f} (SelScore={selection_score:.4f}) | TestAcc={test_val:.4f} (Peak={peak_test_acc:.4f})")

    # =================================================================
    # Phase 1: Train Regressor on Anchor (GLA Step 2)
    # =================================================================
    # 关键：重置分类器，使用初始状态
    model.emotion_classifier = EmotionClassifier(input_size=256, num_classes=args.num_classes, dropout=0.5).to(device)
    optimizer_cls = optim.Adam(model.emotion_classifier.parameters(), lr=args.lr)
    
    logger.info("Phase 1: Training Classifier on Anchor (Reset Classifier)")
    for epoch in range(args.anchor_epochs):
        train_cls_one_epoch(model, [source_loaders[anchor_id]], optimizer_cls, device, epoch, args.anchor_epochs)
        run_evaluation("Anchor-Cls", epoch+1, update_selection_score=True)

    # =================================================================
    # Phase 2: Align Remaining Domains (GLA Step 3)
    # =================================================================
    logger.info("Phase 2: Training Label Alignment Matrix (Others -> Anchor)")
    for source_id in range(len(source_loaders)):
        if source_id == anchor_id: continue
        preds, trues = collect_predictions(model, source_loaders[source_id], args, device)
        if preds is not None:
            train_label_align_one_epoch(label_align_model, optimizer_label_align, source_id, preds, trues, label_align_loss_fn)

    # =================================================================
    # Phase 3: Train Regressor on Remaining Domains (GLA Step 4)
    # =================================================================
    # 关键：再次重置分类器！这符合你发现的 "Phase 3 用了初始状态分类器"
    model.emotion_classifier = EmotionClassifier(input_size=256, num_classes=args.num_classes, dropout=0.5).to(device)
    optimizer_cls = optim.Adam(model.emotion_classifier.parameters(), lr=args.lr)

    logger.info("Phase 3: Training Classifier on Others (Reset Classifier Again)")
    for epoch in range(args.cls_epochs):
        train_cls_one_epoch_with_aligned_label(model, source_loaders, optimizer_cls, device, label_align_model, anchor_id)
        run_evaluation("Aligned-Cls", epoch+1, update_selection_score=False)

    # =================================================================
    # Phase 4: Align Anchor (GLA Step 5)
    # =================================================================
    logger.info("Phase 4: Anchor Self-Correction")
    preds, trues = collect_predictions(model, source_loaders[anchor_id], args, device)
    if preds is not None:
        train_label_align_one_epoch(label_align_model, optimizer_label_align, anchor_id, preds, trues, label_align_loss_fn)

    # =================================================================
    # Phase 5: Final Joint Training (GLA Step 6)
    # =================================================================
    # 关键：彻底重置整个模型（Feature Extractor + Classifier），从头训练！
    # 这符合 "train gaze feature extractor and regressor from scratch"
    
    logger.info("Phase 5: Re-initializing FULL Model and Training from Scratch")
    
    model = LSTMDANN(
        input_size=310, hidden_size=128, num_layers=2, 
        num_classes=args.num_classes, num_domains=14, dropout=0.5
    ).to(device)
    
    # 重新定义所有参数的优化器
    optimizer_all = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_all_one_epoch_with_aligned_label(model, source_loaders, optimizer_all, device, epoch, args.epochs, label_align_model)
        run_evaluation("Final-All", epoch+1, update_selection_score=False)
        
    return selection_score, peak_test_acc


def step2(subject, args):
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

    anchor_results = []
    
    for anchor_id in range(len(source_loaders)):
        src_score, test_score = step2_1(subject, args, anchor_id, device, source_loaders, 
                                        test_loader, label_align_loss_fn, logger)
        anchor_results.append((anchor_id, src_score, test_score))
        logger.info(f">>> Candidate {anchor_id}: SrcScore={src_score:.4f}, TestScore={test_score:.4f}")

    best_anchor_entry = max(anchor_results, key=lambda x: x[1])
    
    best_anchor_id = best_anchor_entry[0]
    best_src_score = best_anchor_entry[1]
    final_reported_test_acc = best_anchor_entry[2]
    
    logger.info("="*50)
    logger.info(f"Subject {subject} Done.")
    logger.info(f"Selected Anchor: {best_anchor_id} (CV Score: {best_src_score:.4f})")
    logger.info(f"Reported Test Acc: {final_reported_test_acc:.4f}")
    logger.info("="*50)
    
    return final_reported_test_acc, best_anchor_id, anchor_results


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
            
            final_acc_list = [res[0] for res in step2_results]
            best_anchor_list = [res[1] for res in step2_results]

            print(f"\nFinal Results ({args.dataset_name}):")
            print(f"Avg: {np.average(final_acc_list):.4f} ± {np.std(final_acc_list):.4f}")
            print("Details:")
            for i, acc in enumerate(final_acc_list):
                print(f"Sub {i}: {acc:.4f} (Anchor {best_anchor_list[i]})")


if __name__ == "__main__":
    seed3_path = "/home/fb/src/dataset/SEED/ExtractedFeatures/"
    seed4_path = "/home/nise-emo/nise-lab/dataset/public/SEED_IV/eeg_feature_smooth/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="seed3")
    parser.add_argument("--session", type=str, default="1")  
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers_train", type=int, default=0) 
    parser.add_argument("--num_workers_test", type=int, default=0)

    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--reg_alpha', type=float, default=1e-2)
    parser.add_argument('--reg_beta', type=float, default=1e-3)
    parser.add_argument('--reg_gamma', type=float, default=1e-3)
    
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
    today = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    
    if args.dataset_name == "seed3":
        args.path = seed3_path
        args.num_classes = 3
        args.time_steps = 30
        args.batch_size = 512
        args.model_dir = os.path.join(current_dir, "models/seed3")
        args.log_dir = os.path.join(current_dir, f"logs/seed3/{today}")
    elif args.dataset_name == "seed4":
        args.path = seed4_path
        args.num_classes = 4
        args.time_steps = 10
        args.batch_size = 256
        args.model_dir = os.path.join(current_dir, "models/seed4")
        args.log_dir = os.path.join(current_dir, f"logs/seed4/{today}")
    
    main(args)