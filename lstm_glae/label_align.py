# lstm_glae/label_align.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelAlignHead(nn.Module):
    """
    为每个训练子域(被试)学习一个 CxC 的标签变换矩阵 T_j (行归一化, 行和=1)。
    训练时: y_soft = onehot @ T_j
    推理时: 不使用该模块 (纯DG设定)。
    """
    def __init__(self, num_domains: int, num_classes: int, tau: float = 1.0, init_eye_scale: float = 5.0):
        """
        num_domains: 训练时的子域数量 (LOSO下= subjects-1)
        num_classes: 类别数 (SEED=3, SEED-IV=4)
        tau: softmax 温度 (越小越尖锐, 0.5~1.0)
        init_eye_scale: 用于初始化对角更大, 让T初始接近单位阵
        """
        super().__init__()
        self.num_domains = num_domains
        self.C = num_classes
        self.tau = tau

        # W_j 形状: [num_domains, C, C]
        self.W = nn.Parameter(torch.zeros(num_domains, self.C, self.C))
        with torch.no_grad():
            eye = torch.eye(self.C).unsqueeze(0).repeat(num_domains, 1, 1)  # [D,C,C]
            self.W.add_(init_eye_scale * eye)  # 对角偏大 -> 初始 T 接近 I

    def get_T(self, domain_id: int):
        """
        返回第 domain_id 个子域的变换矩阵 T_j (行归一化, 行和=1)
        """
        # softmax over last dim (行归一化)
        Tj = F.softmax(self.W[domain_id] / self.tau, dim=-1)  # [C, C], 每行和=1
        return Tj

    def forward(self, domain_id: int):
        return self.get_T(domain_id)

    @staticmethod
    def identity_reg(Tj: torch.Tensor):
        """ ||T - I||_F^2 """
        C = Tj.size(0)
        I = torch.eye(C, device=Tj.device, dtype=Tj.dtype)
        return torch.norm(Tj - I, p='fro') ** 2

    @staticmethod
    def offdiag_reg(Tj: torch.Tensor):
        """ 非对角元素的平方和，鼓励主要沿对角附近对齐 """
        C = Tj.size(0)
        off = Tj - torch.diag(torch.diag(Tj))
        return (off ** 2).sum()

    @staticmethod
    def row_entropy(Tj: torch.Tensor, eps: float = 1e-8):
        """
        行熵之和: sum_i H(T_i,*)
        惩罚该项可降低熵(更尖锐)，防止均匀塌缩。
        """
        P = Tj.clamp_min(eps)
        return -(P * torch.log(P)).sum()
