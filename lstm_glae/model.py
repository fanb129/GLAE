# lstm_glae/model.py
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=310, hidden_size=128, num_layers=2, num_classes=3, dropout=0.5, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * mult, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)  # 便于Soft-CE计算

    def forward(self, x):
        # x: [B, T, 310]
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # 取最后一步
        logits = self.fc(out)         # [B, C]
        log_probs = self.logsoftmax(logits)
        return log_probs              # 输出log p
