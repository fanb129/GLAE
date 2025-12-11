import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# Gradient Reversal Layer
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class LSTMFeatureExtractor(nn.Module):
    """LSTM特征提取器"""
    def __init__(self, input_size=310, hidden_size=128, num_layers=2, dropout=0.5):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
    def forward(self, x):
        # x: (batch, time_steps, 310)
        out, _ = self.lstm(x)
        # 取最后时间步的hidden state
        features = out[:, -1, :]  # (batch, hidden_size * 2)
        return features


class EmotionClassifier(nn.Module):
    """情感分类器"""
    def __init__(self, input_size=256, num_classes=3, dropout=0.5):
        super(EmotionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, features):
        output = self.classifier(features)
        return F.log_softmax(output, dim=1)


class DomainClassifier(nn.Module):
    """域分类器（用于DANN）"""
    def __init__(self, input_size=256, num_domains=14, dropout=0.5):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_domains)
        )
        
    def forward(self, features, alpha):
        # 使用梯度反转层
        reverse_features = ReverseLayerF.apply(features, alpha)
        output = self.classifier(reverse_features)
        return F.log_softmax(output, dim=1)


class LSTMDANN(nn.Module):
    """完整的LSTM+DANN模型"""
    def __init__(self, 
                 input_size=310,
                 hidden_size=128,
                 num_layers=2,
                 num_classes=3,
                 num_domains=14,
                 dropout=0.5):
        super(LSTMDANN, self).__init__()
        
        # 特征提取器
        self.feature_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 情感分类器
        self.emotion_classifier = EmotionClassifier(
            input_size=hidden_size * 2,  # 双向LSTM
            num_classes=num_classes,
            dropout=dropout
        )
        
        # 域分类器
        self.domain_classifier = DomainClassifier(
            input_size=hidden_size * 2,
            num_domains=num_domains,
            dropout=dropout
        )
        
    def forward(self, x, alpha=0.0):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch, time_steps, input_size)
            alpha: 梯度反转层的参数，控制域适应的强度
            
        Returns:
            emotion_output: 情感分类输出
            domain_output: 域分类输出
            features: 提取的特征
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 情感分类
        emotion_output = self.emotion_classifier(features)
        
        # 域分类（使用梯度反转）
        domain_output = self.domain_classifier(features, alpha)
        
        return emotion_output, domain_output, features


class LSTMOnly(nn.Module):
    """仅用于测试的LSTM模型（不带DANN）"""
    def __init__(self, 
                 input_size=310,
                 hidden_size=128,
                 num_layers=2,
                 num_classes=3,
                 dropout=0.5):
        super(LSTMOnly, self).__init__()
        
        self.feature_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.emotion_classifier = EmotionClassifier(
            input_size=hidden_size * 2,
            num_classes=num_classes,
            dropout=dropout
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        emotion_output = self.emotion_classifier(features)
        return emotion_output, features
    

class LabelAlignLoss(nn.Module):
    """标签对齐损失函数"""
    def __init__(self, num_classes, reg_alpha=1e-2, reg_beta=1e-3, reg_gamma=1e-3):
        super(LabelAlignLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_alpha = reg_alpha
        self.reg_beta = reg_beta
        self.reg_gamma = reg_gamma
    
    def forward(self, aligned_probs, target_probs, T_matrices):
        """
        对齐损失计算
        """
        # KL散度损失
        # kl_loss = F.kl_div(torch.log(aligned_probs + 1e-8), target_probs, reduction='batchmean')
        kl_loss = F.kl_div(torch.log(aligned_probs + 1e-6), target_probs + 1e-6, reduction='batchmean')
        # 正则化项
        reg_loss = 0
        for T in T_matrices:
            # 单位矩阵正则化
            identity_reg = torch.norm(T - torch.eye(self.num_classes, device=T.device), p='fro') ** 2
            
            # 行熵正则化
            row_entropy = -(T * torch.log(T + 1e-8)).sum(dim=-1).mean()
            
            # 非对角线正则化
            off_diag = T - torch.diag(torch.diag(T))
            off_diag_reg = torch.norm(off_diag, p='fro') ** 2
            
            reg_loss += self.reg_alpha * identity_reg + self.reg_beta * row_entropy + self.reg_gamma * off_diag_reg
        
        return kl_loss + reg_loss
    

# 标签对齐
class LabelAlignModel(nn.Module):
    def __init__(self, num_domains, num_classes, tau=0.7, init_eye_scale=5.0):
        super(LabelAlignModel, self).__init__()
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.tau = tau
        
        # 为每个域学习一个C×C的标签变换矩阵
        self.T = nn.Parameter(torch.zeros(num_domains, num_classes, num_classes))
        
        # 初始化为单位矩阵的缩放版本
        with torch.no_grad():
            eye = torch.eye(num_classes).unsqueeze(0).repeat(num_domains, 1, 1)
            self.T.add_(init_eye_scale * eye)
    
    def get_T(self, domain_id):
        """获取指定域的标签变换矩阵"""
        T_j = F.softmax(self.T[domain_id] / self.tau, dim=-1)  # 行归一化
        return T_j
    
    def forward(self, labels,domain_id):
        """
        对标签进行对齐变换
        labels: one-hot编码的标签 [batch_size, num_classes]
        domain_id: 域索引
        """
        T_j = self.get_T(domain_id)
        aligned_labels = torch.matmul(labels, T_j)
        return aligned_labels
    
    def get_aligned_label(self, y, domain_id):
        y_onehot = torch.zeros(y.size(0), self.num_classes, device=y.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        T_domain = self.get_T(domain_id)
        aligned_labels = torch.matmul(y_onehot, T_domain)
        
        return aligned_labels