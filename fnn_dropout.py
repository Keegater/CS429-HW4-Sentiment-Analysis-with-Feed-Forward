
import torch
import torch.nn as nn

class FNN_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_prob=None):
        super(FNN_Dropout, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
        



    def forward(self, x):
        return self.net(x)
