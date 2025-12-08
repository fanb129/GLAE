import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=310, hidden_size=128, num_layers=2, num_classes=3, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: (batch, time_steps, 310)
        out, _ = self.lstm(x)  
        out = out[:, -1, :]  # 取最后时间步的hidden state
        out = self.fc(out)
        return self.softmax(out)
