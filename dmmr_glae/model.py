import torch
import torch.nn as nn
import torch.nn.functional as F
from GradientReverseLayer import ReverseLayerF
import random
import copy

# The ABP module
class Attention(nn.Module):
    def __init__(self, cuda, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        if cuda:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim).cuda())
            self.u_linear = nn.Parameter(torch.randn(input_dim).cuda())
        else:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
            self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size, time_steps):
        x_reshape = torch.Tensor.reshape(x, [-1, self.input_dim])
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear)+ self.u_linear,1)
        res = torch.mul(attn_softmax, x_reshape)
        res = torch.Tensor.reshape(res, [batch_size, time_steps, self.input_dim])
        return res

class LSTM(nn.Module):
    def __init__(self, input_dim=310, output_dim=64, layers=2, location=-1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=layers, batch_first=True)
        self.location = location
    def forward(self, x):
        # self.lstm.flatten_parameters()
        feature, (hn, cn) = self.lstm(x)
        return feature[:, self.location, :], hn, cn

class Encoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2):
        super(Encoder, self).__init__()
        self.theta = LSTM(input_dim, hid_dim, n_layers)
    def forward(self, x):
        x_h = self.theta(x)
        return x_h

class Decoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2,output_dim=310):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden, cell, time_steps):
        out =[]
        out1 = self.fc_out(input)
        out.append(out1)
        out1= out1.unsqueeze(0)  # input = [batch size] to [1, batch size]
        for i in range(time_steps-1):
            output, (hidden, cell) = self.rnn(out1,
                                              (hidden, cell))  # output =[seq len, batch size, hid dim* ndirection]
            out_cur = self.fc_out(output.squeeze(0))  # prediction = [batch size, output dim]
            out.append(out_cur)
            out1 = out_cur.unsqueeze(0)
        out.reverse()
        out = torch.stack(out)
        out = out.transpose(1,0) #batch first
        return out, hidden, cell


#namely The Subject Classifier SD
class DomainClassifier(nn.Module):
    def __init__(self, input_dim =64, output_dim=14):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.classifier(x)
        return x

# The MSE loss
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


# proposed DMMR model
class DMMRPreTrainingModel(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(DMMRPreTrainingModel, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        # Noise Injection, with the proposed method Time Steps Shuffling
        # x = timeStepsShuffle(x)
        # The ABP module
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        # Encoder the weighted features with one-layer LSTM
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)
        # The DG_DANN module
        # The GRL layer in the first stage
        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        # The Subject Discriminator
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        # The domain adversarial loss
        sim_loss = F.nll_loss(subject_predict, subject_id)

        # Build Supervision for Decoders, the inputs are also weighted
        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            # Reconstruct features in the first stage
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            # The proposed mix method for data augmentation
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            # Reconstruct features in the second stage
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            # Compute the reconstructive loss in the second stage only
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss


class DMMRFineTuningModel(nn.Module):
    def __init__(self, cuda, baseModel, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(DMMRFineTuningModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        # The ABP module and sharedEncoder are from the pretrained model
        self.attentionLayer = self.baseModel.attentionLayer
        self.sharedEncoder = self.baseModel.sharedEncoder
        # Add a new emotion classifier for emotion recognition
        self.cls_fc = nn.Sequential(nn.Linear(64, 64, bias=False), nn.BatchNorm1d(64),
                               nn.ReLU(inplace=True), nn.Linear(64, number_of_category, bias=True))
        self.mse = MSE()
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, label_src=0):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)
        x_logits = self.cls_fc(shared_last_out)
        x_pred = F.log_softmax(x_logits, dim=1)
        cls_loss = F.nll_loss(x_pred, label_src.squeeze())
        return x_pred, x_logits, cls_loss

class DMMRTestModel(nn.Module):
    def __init__(self, baseModel):
        super(DMMRTestModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
    def forward(self, x):
        x = self.baseModel.attentionLayer(x, self.baseModel.batch_size, self.baseModel.time_steps)
        shared_last_out, shared_hn, shared_cn = self.baseModel.sharedEncoder(x)
        x_shared_logits = self.baseModel.cls_fc(shared_last_out)
        return x_shared_logits


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