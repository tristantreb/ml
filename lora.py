import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        # Frozen pre-trained weights
        self.W = nn.Linear(in_dim, out_dim)
        self.W.weight.requires_grad = False

        # Low-rank matrics
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim)) # Initialised to 0
        self.scaling = alpha / rank

    def forward(self, x):
        # Can write W(x) because nn.Linear is a subcalss of nn.Module
        # Can't write B(x) because B is a nn.Parameter
        return self.W(x) + self.scaling * (x @ self.A @ self.B)
    