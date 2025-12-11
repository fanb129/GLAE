import os, sys, argparse,random
import numpy as np
import torch
import torch.optim as optim
import copy
import time
import logging
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.preprocess import getDataLoaders
from train_glae import train_all_one_epoch, train_cls_one_epoch, evaluate, train_label_align_one_epoch, train_cls_one_epoch_with_aligned_label, train_all_one_epoch_with_aligned_label
from model import LSTMDANN, LabelAlignModel, LabelAlignLoss


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_pretrain_logging(subject, log_dir):
    """
    为每个受试者设置单独的日志文件。
    """
    log_file = os.path.join(log_dir, f"pretrain_subject_{subject}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler()  # 仍然在控制台打印
        ]
    )
    return logging.getLogger()

def setup_logging(subject, log_dir):
    """
    为每个受试者设置单独的日志文件。
    """
    log_file = os.path.join(log_dir, f"subject_{subject}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler()  # 仍然在控制台打印
        ]
    )
    return logging.getLogger()

def train_label_align_model(args,source_loader,device,model,label_align_model,optimizer_label_align,source_id,label_align_loss_fn,subject,anchor_id):
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for x, y in source_loader:
            x, y = x.to(device), y.squeeze().to(device)
            soft_probs, _, _ = model(x)
            # 转换为one-hot真实标签
            y_onehot = torch.zeros(y.size(0), args.num_classes, device=y.device)
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            
            all_predictions.append(soft_probs)
            all_true_labels.append(y_onehot)
        predictions = torch.cat(all_predictions, dim=0) 
        true_labels = torch.cat(all_true_labels, dim=0)
    for epoch in range(args.align_epochs):
        la_loss = train_label_align_one_epoch(label_align_model, optimizer_label_align, source_id, predictions, true_labels, label_align_loss_fn)
        # logger.info(f"labelAlign [Subject {subject}, Epoch {epoch+1}/{args.align_epochs}] Anchor Domain={anchor_id} Target Domain={source_id} Label Align Loss={la_loss:.4f} ")


def step1(subject, args):
    # 设置当前进程的日志
    logger = setup_pretrain_logging(subject, args.log_dir)
    logger.info(f"=== PreTraining, leaving subject {subject} for test ===")
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
    # 训练所有参数
    optimizer_all = optim.Adam(model.parameters(), lr=args.lr)

    final_acc = 0.0
    best_model = None
    for epoch in range(args.epochs_pretrain):
        train_loss, train_acc = train_all_one_epoch(model, source_loaders, optimizer_all, device, epoch, args.epochs_pretrain)
        test_acc = evaluate(model, test_loader, device)
        if test_acc > final_acc:
            final_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
        logger.info(f"pretrain [Subject {subject}, Epoch {epoch+1}/{args.epochs_pretrain}] Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} Best Acc={final_acc:.4f}")
    
    torch.save(best_model, os.path.join(args.model_dir, f"pretrain_model_subject{subject}.pth"))
    return final_acc


def step2(subject, args):
    logger = setup_logging(subject, args.log_dir)
    logger.info(f"=== GLAE, leaving subject {subject} for test ===")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_loaders, test_loader = getDataLoaders(subject, args)
    label_align_loss_fn = LabelAlignLoss(num_classes=args.num_classes, reg_alpha=args.reg_alpha, reg_beta=args.reg_beta, reg_gamma=args.reg_gamma)

    acc_list = []
    final_acc = 0.0
    best_anchor = 0
    # with multiprocessing.Pool(processes=14) as pool:
    #     anchors = range(14)
    #     acc_list = pool.starmap(step2_1, [(subject, args, anchor_id, device, source_loaders, test_loader, label_align_loss_fn) for anchor_id in anchors])
    for anchor_id in range(len(source_loaders)):
        acc = step2_1(subject, args, anchor_id, device, source_loaders, test_loader, label_align_loss_fn, logger)
        acc_list.append(acc)
    final_acc = max(acc_list)
    best_anchor = acc_list.index(final_acc)
    return final_acc, best_anchor, acc_list
    
    
def step2_1(subject, args, anchor_id, device, source_loaders, test_loader, label_align_loss_fn, logger):
    # logger = setup_logging(subject, anchor_id, args.log_dir)
    logger.info(f"=========================Subject {subject}, Anchor {anchor_id}===========================")
    # 加载预训练模型
    model = LSTMDANN(
        input_size=310,
        hidden_size=128,
        num_layers=2,
        num_classes=args.num_classes,
        num_domains=14,
        dropout=0.5).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, f"pretrain_model_subject{subject}.pth")))
    # 训练所有参数
    optimizer_all = optim.Adam(model.parameters(), lr=args.lr)
    # 只训练分类器
    optimizer_cls = optim.Adam(model.emotion_classifier.parameters(), lr=args.lr)
    
    label_align_model = LabelAlignModel(
        num_domains=len(source_loaders), 
        num_classes=args.num_classes, 
        tau=args.tau).to(device)
    optimizer_label_align = torch.optim.Adam(label_align_model.parameters(), lr=args.lr)
    final_acc = 0.0
    # 训练anchor的分类器
    for epoch in range(args.anchor_epochs):
        train_loss, train_acc = train_cls_one_epoch(model, [source_loaders[anchor_id]], optimizer_cls, device, epoch, args.anchor_epochs)
        test_acc = evaluate(model, test_loader, device)
        if test_acc > final_acc:
            final_acc = test_acc
        logger.info(f"anchor [Subject {subject}, Anchor {anchor_id}, Epoch {epoch+1}/{args.anchor_epochs}] Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} Best Acc={final_acc:.4f}")
    # 用anchor的分类器预测其他源域数据,并训练标签对齐网络
    for source_id in range(len(source_loaders)):
        if source_id == anchor_id:
            continue
        source_loader = source_loaders[source_id]
        logger.info(f"labelAlign [Subject {subject}, Anchor {anchor_id} Target {source_id}] Start")
        train_label_align_model(args,source_loader,device,model,label_align_model,optimizer_label_align,source_id,label_align_loss_fn,subject,anchor_id)

    #利用对齐之后的标签训练model的分类器
    for epoch in range(args.cls_epochs):
        train_loss, train_acc = train_cls_one_epoch_with_aligned_label(model, source_loaders, optimizer_cls, device, label_align_model, anchor_id)
        test_acc = evaluate(model, test_loader, device)
        if test_acc > final_acc:
            final_acc = test_acc
        logger.info(f"cls [Subject {subject}, Anchor {anchor_id}, Epoch {epoch+1}/{args.cls_epochs}] Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} Best Acc={final_acc:.4f}")
    # 利用model预测anchor，对齐anchor的标签
    anchor_loader = source_loaders[anchor_id]
    logger.info(f"labelAlign [Subject {subject}, Anchor {anchor_id} Target {anchor_id}] Start")
    train_label_align_model(args,anchor_loader,device,model,label_align_model,optimizer_label_align,anchor_id,label_align_loss_fn,subject,anchor_id)
    # 利用所有对齐后的标签重新训练model的所有参数
    for epoch in range(args.epochs):
        train_loss, train_acc = train_all_one_epoch_with_aligned_label(model, source_loaders, optimizer_all, device, epoch, args.epochs, label_align_model)
        test_acc = evaluate(model, test_loader, device)
        if test_acc > final_acc:
            final_acc = test_acc
        logger.info(f"all [Subject {subject}, Anchor {anchor_id}, Epoch {epoch+1}/{args.epochs}] Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} Best Acc={final_acc:.4f}")
    return final_acc


def main(args):
    # subject = args.subject
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    best_anchor_list = []
    anchor_acc_list = []
    step1_acc_list = []
    step2_acc_list = []
    # if args.step1:
    #     acc_step1 = step1(subject, args)
    # if args.step2:
    #     acc_step2, best_anchor, acc_list = step2(subject, args)
    # print(f"Subject {subject}, Step1 Acc={acc_step1:.4f}, Step2 Acc={acc_step2:.4f}, Best Anchor={best_anchor}")
    # for i in range(len(acc_list)):
    #     print(f"Subject {subject}, Anchor {i}, Acc={acc_list[i]:.4f}")
    # 使用 multiprocessing 的 Pool 来并行运行15个 step1
    with multiprocessing.Pool(processes=15) as pool:
        subjects = range(15)
        if args.step1:
            step1_acc_list = pool.starmap(step1, [(subject, args) for subject in subjects])
            if args.dataset_name == "seed3":
                print("SEED Pretrain")
            elif args.dataset_name == "seed4":
                print("SEED IV Pretrain")
            print("final acc avg:", str(np.average(step1_acc_list)))
            print("final acc std:", str(np.std(step1_acc_list)))
            print('final each acc:')
            # 打印索引和值
            for i,acc in enumerate(step1_acc_list):
                print(f'Subject {i}: {acc:.4f}')
        if args.step2:
            step2_result_list = pool.starmap(step2, [(subject, args) for subject in subjects])
            # print(step2_acc_list)
            for i, (final_acc, best_anchor, acc_list) in enumerate(step2_result_list):
                step2_acc_list.append(final_acc)
                best_anchor_list.append(best_anchor)
                anchor_acc_list.append(acc_list)
            if args.dataset_name == "seed3":
                print("SEED 最终结果")
            elif args.dataset_name == "seed4":
                print("SEED IV 最终结果")
            print("final acc avg:", str(np.average(step2_acc_list)))
            print("final acc std:", str(np.std(step2_acc_list)))
            print('final each acc:')
            # 打印索引和值
            for i,acc in enumerate(step2_acc_list):
                print(f'Subject {i}: {acc:.4f}')
                print(f'Best Anchor: {best_anchor_list[i]}')
                for j, acc in enumerate(anchor_acc_list[i]):
                    print(f'Subject {i}, Anchor {j}: {acc:.4f}')

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    seed3_path = "/home/fb/src/dataset/SEED/ExtractedFeatures/"
    seed4_path = "/home/nise-emo/nise-lab/dataset/public/SEED_IV/eeg_feature_smooth/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="seed4")
    parser.add_argument("--session", type=str, default="1")  
    parser.add_argument("--num_workers_train", type=int, default=0)
    parser.add_argument("--num_workers_test", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    # 标签对齐参数
    parser.add_argument('--tau', type=float, default=0.7, help='标签对齐softmax温度参数')
    parser.add_argument('--reg_alpha', type=float, default=1e-2, help='标签对齐正则化系数')
    parser.add_argument('--reg_beta', type=float, default=1e-3, help='行熵正则化系数')
    parser.add_argument('--reg_gamma', type=float, default=1e-3, help='非对角线惩罚系数')
    # 训练参数
    parser.add_argument('--epochs_pretrain', type=int, default=300)
    parser.add_argument('--align_epochs', type=int, default=150)
    parser.add_argument('--anchor_epochs', type=int, default=150)
    parser.add_argument('--cls_epochs', type=int, default=150)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument('--epochs_pretrain', type=int, default=1)
    # parser.add_argument('--align_epochs', type=int, default=1)
    # parser.add_argument('--anchor_epochs', type=int, default=1)
    # parser.add_argument('--cls_epochs', type=int, default=1)
    # parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--cuda", type=str, default='2')
    parser.add_argument("--step1", type=bool, default=False)
    parser.add_argument("--step2", type=bool, default=True)

    
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
    else:
        print("Invalid dataset name")
        exit()
    main(args)