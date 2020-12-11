import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=11, out_features=150)
        self.fc2 = nn.Linear(in_features=150, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=3)

    
    def forward(self,t):
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.out(t)
        return t  