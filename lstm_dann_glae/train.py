import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_alpha(epoch, total_epochs, batch_idx, total_batches):
    """计算 DANN 中的 GRL 参数 alpha (0 -> 1)"""
    p = float(batch_idx + epoch * total_batches) / (total_epochs * total_batches)
    return 2. / (1. + np.exp(-10 * p)) - 1

def train_all_one_epoch(model, loaders, optimizer, device, epoch, epochs):
    """
    Step 1 Pretrain: 同时训练特征提取、情感分类和域判别
    """
    iteration = len(loaders[0]) 
    # 创建所有源域的迭代器
    source_iters = [iter(loader) for loader in loaders]
    
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for i in range(iteration):
        # 计算当前的 alpha
        alpha = get_alpha(epoch, epochs, i, iteration)
        
        for domain_id, loader_iter in enumerate(source_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                # 如果某个loader数据耗尽（通常不会，因为长度取了min或max），重置
                source_iters[domain_id] = iter(loaders[domain_id])
                x, y = next(source_iters[domain_id])
            
            x, y = x.to(device), y.squeeze().to(device)
            
            # 域标签
            domain_label = torch.full((x.size(0),), domain_id, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            # Forward
            emotion_output, domain_output, _ = model(x, alpha)
            
            # Losses
            cls_loss = F.nll_loss(emotion_output, y)
            dann_loss = F.nll_loss(domain_output, domain_label)
            
            # Total Loss (0.05 是 DANN 的常见权重，可调)
            loss = cls_loss + 0.05 * dann_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
    return total_loss / total, correct / total


def train_cls_one_epoch(model, loaders, optimizer, device, epoch, epochs):
    """
    Step 2 Phase 1: 仅训练分类器 (用于 Anchor 预热)
    """
    model.train()
    # 冻结特征提取器，只训练分类头
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    
    total_loss, correct, total = 0, 0, 0
    
    for loader in loaders: # 通常这里只传入 [anchor_loader]
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            
            optimizer.zero_grad()
            # alpha=0 因为不关心域判别
            emotion_output, _, _ = model(x, 0)
            
            loss = F.nll_loss(emotion_output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
    # 解冻 (为后续步骤做准备)
    for param in model.feature_extractor.parameters():
        param.requires_grad = True
        
    return total_loss / total, correct / total


# def train_label_align_one_epoch(label_align_model, optimizer, source_id, 
#                                 predictions, true_labels, label_align_loss_fn):
#     """
#     Step 2 Phase 2: 训练标签对齐矩阵 T
#     Args:
#         predictions: 模型对源域数据的预测 (Logits 或 LogSoftmax)
#         true_labels: 真实标签 (One-hot)
#     """
#     label_align_model.train()
#     optimizer.zero_grad()
    
#     # 获取当前域的变换矩阵 T
#     T_target = label_align_model.get_T(source_id)
    
#     # 计算对齐后的标签分布: Y_aligned = Y_true * T
#     # true_labels 应该是 one-hot
#     aligned_labels = torch.matmul(true_labels, T_target)

#     # 处理模型预测值
#     # 注意: LSTMDANN 输出是 log_softmax
#     # LabelAlignLoss 中的 target_probs 需要是概率分布 (0-1)
#     target_probs = torch.exp(predictions)
    
#     # 计算损失
#     # Loss = KL(log(aligned), target) + Regularization
#     loss = label_align_loss_fn(aligned_labels, target_probs, [T_target])
    
#     loss.backward()
#     optimizer.step()
#     return loss.item()

def train_label_align_one_epoch(label_align_model, optimizer, source_id, 
                                predictions, true_labels, label_align_loss_fn):
    """
    Step 2 Phase 2: 训练标签对齐矩阵 T
    """
    label_align_model.train()
    optimizer.zero_grad()
    
    T_target = label_align_model.get_T(source_id)
    
    # 监控 T 矩阵：计算对角线元素的平均值
    # 如果该值接近 1.0，说明 T 是单位矩阵（不改变标签）
    # 如果该值很低（如 <0.5），说明标签被严重打乱
    with torch.no_grad():
        diag_mean = torch.diagonal(T_target).mean().item()
    
    aligned_labels = torch.matmul(true_labels, T_target)
    target_probs = torch.exp(predictions) # LogSoftmax -> Probs
    
    loss = label_align_loss_fn(aligned_labels, target_probs, [T_target])
    
    loss.backward()
    optimizer.step()
    
    # 返回 loss 和 对角线均值 用于打印
    return loss.item(), diag_mean


def train_cls_one_epoch_with_aligned_label(model, loaders, optimizer, device, 
                                           label_align_model, anchor_id):
    """
    Step 2 Phase 3: 使用对齐后的标签微调分类器
    """
    model.train()
    # 冻结特征提取器
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
        
    total_loss, correct, total = 0, 0, 0
    
    for domain_id, loader in enumerate(loaders):
        # Anchor 域通常保持原标签 (或在 Phase 4 自对齐，这里暂跳过或包含取决于策略)
        # GLA论文策略：Anchor单独训练或最后一起训练。
        # 这里我们对非Anchor域使用对齐标签
        
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            
            # 获取对齐后的软标签
            with torch.no_grad():
                # 如果是 Anchor，T 应该是接近单位矩阵，或者我们在外面控制是否跳过
                # 这里假设传入的 label_align_model 已经包含了所有域的 T
                aligned_labels = label_align_model.get_aligned_label(y, domain_id)
            
            optimizer.zero_grad()
            emotion_output, _, _ = model(x, 0)
            
            # Cross Entropy with Soft Labels
            # emotion_output 是 log_softmax
            # Loss = - sum(target * log_pred)
            loss = -(aligned_labels * emotion_output).sum(dim=1).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    # 解冻
    for param in model.feature_extractor.parameters():
        param.requires_grad = True
        
    return total_loss / total, correct / total


def train_all_one_epoch_with_aligned_label(model, loaders, optimizer, device, 
                                           epoch, epochs, label_align_model):
    """
    Step 2 Phase 5: 使用对齐后的标签，全量微调 (Feature + Cls + DANN)
    """
    iteration = len(loaders[0])
    source_iters = [iter(loader) for loader in loaders]
    
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for i in range(iteration):
        alpha = get_alpha(epoch, epochs, i, iteration)
        
        for domain_id, loader_iter in enumerate(source_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                source_iters[domain_id] = iter(loaders[domain_id])
                x, y = next(source_iters[domain_id])
                
            x, y = x.to(device), y.squeeze().to(device)
            domain_label = torch.full((x.size(0),), domain_id, dtype=torch.long, device=device)

            # 获取对齐后的标签
            with torch.no_grad():
                aligned_labels = label_align_model.get_aligned_label(y, domain_id)

            optimizer.zero_grad()
            
            emotion_output, domain_output, _ = model(x, alpha)
            
            # Losses
            dann_loss = F.nll_loss(domain_output, domain_label)
            # Soft Label Loss
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
    """验证模型准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze().to(device)
            emotion_output, _, _ = model(x)
            _, predicted = torch.max(emotion_output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0