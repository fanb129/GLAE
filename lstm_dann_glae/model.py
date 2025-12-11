import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ==========================================
# 1. 基础组件 (GRL, LSTM, Classifier)
# ==========================================

class ReverseLayerF(Function):
    """梯度反转层 (Gradient Reversal Layer)"""
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
        # x: (batch, time_steps, input_size)
        # out: (batch, time_steps, hidden_size*2)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出作为序列特征
        # 对于双向LSTM，out[:, -1, :] 包含了前向的最后一步和后向的第一步（即最后处理的一步）
        features = out[:, -1, :]  
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
        # 注意：这里返回的是 log_softmax，适用于 NLLLoss
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
        # 使用梯度反转
        reverse_features = ReverseLayerF.apply(features, alpha)
        output = self.classifier(reverse_features)
        return F.log_softmax(output, dim=1)

# ==========================================
# 2. 完整模型 (LSTMDANN)
# ==========================================

class LSTMDANN(nn.Module):
    """
    完整的 LSTM + DANN 模型架构
    包含特征提取、情感分类和域判别
    """
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
        
        # 情感分类器 (输入维度是 hidden_size * 2 因为是双向LSTM)
        self.emotion_classifier = EmotionClassifier(
            input_size=hidden_size * 2,
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
        Args:
            x: 输入特征
            alpha: GRL层的反转系数
        Returns:
            emotion_output: 情感分类结果 (LogSoftmax)
            domain_output: 域分类结果 (LogSoftmax)
            features: 提取的中间特征 (用于t-SNE等)
        """
        features = self.feature_extractor(x)
        emotion_output = self.emotion_classifier(features)
        domain_output = self.domain_classifier(features, alpha)
        
        return emotion_output, domain_output, features

# ==========================================
# 3. GLA 标签对齐相关 (AlignModel, AlignLoss)
# ==========================================

class LabelAlignModel(nn.Module):
    """
    标签对齐模型
    学习每个源域相对于 Anchor 域的混淆矩阵 T
    """
    def __init__(self, num_domains, num_classes, tau=0.7, init_eye_scale=5.0):
        super(LabelAlignModel, self).__init__()
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.tau = tau
        
        # T: (num_domains, num_classes, num_classes)
        # 每一个域对应一个混淆矩阵
        self.T = nn.Parameter(torch.zeros(num_domains, num_classes, num_classes))
        
        # 初始化：倾向于单位矩阵 (Identity)
        # init_eye_scale 越大，初始状态越接近不修改标签
        with torch.no_grad():
            eye = torch.eye(num_classes).unsqueeze(0).repeat(num_domains, 1, 1)
            self.T.add_(init_eye_scale * eye)
    
    def get_T(self, domain_id):
        """获取指定域的归一化转移矩阵 (Row-Normalized)"""
        # 使用 softmax 保证矩阵每一行和为1（概率分布）
        T_j = F.softmax(self.T[domain_id] / self.tau, dim=-1)
        return T_j
    
    def get_aligned_label(self, y, domain_id):
        """
        计算对齐后的软标签
        Args:
            y: 真实标签 (LongTensor), shape (batch,)
            domain_id: 当前域的索引
        Returns:
            aligned_labels: 对齐后的软标签分布, shape (batch, num_classes)
        """
        # 转换为 One-Hot
        y_onehot = torch.zeros(y.size(0), self.num_classes, device=y.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        
        # 获取转移矩阵并计算: y_aligned = y_true * T
        T_domain = self.get_T(domain_id)
        aligned_labels = torch.matmul(y_onehot, T_domain)
        
        return aligned_labels

class LabelAlignLoss(nn.Module):
    """
    标签对齐损失函数
    Loss = KL(Model_Pred || Aligned_Label) + Regularization
    """
    def __init__(self, num_classes, reg_alpha=1e-2, reg_beta=1e-3, reg_gamma=1e-3):
        super(LabelAlignLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_alpha = reg_alpha # 单位矩阵约束权重
        self.reg_beta = reg_beta   # 熵约束权重 (Sharpness)
        self.reg_gamma = reg_gamma # 非对角线稀疏权重
    
    def forward(self, aligned_probs, target_probs, T_matrices):
        """
        Args:
            aligned_probs: 对齐后的标签分布 (Q)
            target_probs: 模型预测的概率分布 (P, 通常是 Teacher 的预测)
            T_matrices: 当前涉及的混淆矩阵列表
        """
        # 1. KL散度损失: D_KL(Target || Aligned)
        # input需要是 log-probs, target 是 probs
        # aligned_probs 经过 T 矩阵运算是概率，所以取 log
        # target_probs 必须是概率 (0-1)
        kl_loss = F.kl_div(
            torch.log(aligned_probs + 1e-8), 
            target_probs + 1e-8, 
            reduction='batchmean'
        )
        
        # 2. 正则化项
        reg_loss = 0
        for T in T_matrices:
            # Identity Regularization: T 应该接近单位矩阵
            identity_reg = torch.norm(T - torch.eye(self.num_classes, device=T.device), p='fro') ** 2
            
            # Entropy Regularization: 最小化熵，鼓励 T 矩阵尖锐 (Confident)
            row_entropy = -(T * torch.log(T + 1e-8)).sum(dim=-1).mean()
            
            # Off-diagonal Regularization: 抑制非对角线元素
            off_diag = T - torch.diag(torch.diag(T))
            off_diag_reg = torch.norm(off_diag, p='fro') ** 2
            
            reg_loss += self.reg_alpha * identity_reg + \
                        self.reg_beta * row_entropy + \
                        self.reg_gamma * off_diag_reg
        
        return kl_loss + reg_loss