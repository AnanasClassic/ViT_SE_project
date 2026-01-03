import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, bias=True, dropout=0.0):
        super(MyMLP, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    def __str__(self):
        return f"MyMLP(in_features={self.fc1.in_features}, hidden_features={self.fc1.out_features}, out_features={self.fc2.out_features}, bias={self.fc1.bias is not None})"
    
    def __repr__(self):
        return self.__str__()
    
class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    
    def __str__(self):
        return f"Residual({self.module})"
    
    def __repr__(self):
        return self.__str__()