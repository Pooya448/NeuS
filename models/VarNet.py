import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Embedder import get_embedder


class VarNet(nn.Module):
    def __init__(self, init_val):
        super(VarNet, self).__init__()
        self.variance = nn.Parameter(torch.tensor(init_val))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
