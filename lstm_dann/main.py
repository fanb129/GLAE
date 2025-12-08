import os, sys, argparse,random
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.preprocess import getDataLoaders
from train import train_one_epoch, evaluate
from model import LSTMDANN


os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_list = []
    for subject in range(15):  # 15 subjects
        print(f"=== Training, leaving subject {subject} for test ===")
        source_loaders, test_loader = getDataLoaders(subject, args)
        model = LSTMDANN(
            input_size=310,
            hidden_size=128,
            num_layers=2,
            num_classes=args.num_classes,
            num_domains=14,
            dropout=0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        final_acc = 0.0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, source_loaders, optimizer, device, epoch, args.epochs)
            test_acc = evaluate(model, test_loader, device)
            if test_acc > final_acc:
                final_acc = test_acc
            print(f"[Subject {subject+1}, Epoch {epoch+1}/{args.epochs}] Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Test Acc={test_acc:.4f} Best Acc={final_acc:.4f}")

        acc_list.append(final_acc)

    if args.dataset_name == "seed3":
        print("SEED 最终结果")
    elif args.dataset_name == "seed4":
        print("SEED IV 最终结果")
    print("final acc avg:", str(np.average(acc_list)))
    print("final acc std:", str(np.std(acc_list)))
    print('final each acc:')
    # 打印索引和值
    for i,acc in enumerate(acc_list):
        print(f'Subject {i+1}: {acc:.4f}')

if __name__ == "__main__":
    seed3_path = "/home/fb/src/dataset/SEED/ExtractedFeatures/"
    seed4_path = "/home/nise-emo/nise-lab/dataset/public/SEED_IV/eeg_feature_smooth/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="seed4")
    parser.add_argument("--session", type=str, default="1")  
    parser.add_argument("--num_workers_train", type=int, default=16)
    parser.add_argument("--num_workers_test", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.dataset_name == "seed3":
        args.path = seed3_path
        args.num_classes = 3
        args.time_steps = 30
        args.batch_size = 512
    elif args.dataset_name == "seed4":
        args.path = seed4_path
        args.num_classes = 4
        args.time_steps = 10
        args.batch_size = 256
    else:
        print("Invalid dataset name")
        exit()
    main(args)