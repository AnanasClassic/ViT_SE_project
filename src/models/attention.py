import torch
import torch.nn as nn

class MyAttention(nn.Module):
    _logged_shapes = False

    def __init__(self, dim, n_heads=1, bias=False, dropout=0.0):
        super(MyAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        assert dim % n_heads == 0
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.scale = self.head_dim ** -0.5
        
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, length, n_channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        if not MyAttention._logged_shapes:
            print(f"Attention shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}, A: {attn_probs.shape}")
            MyAttention._logged_shapes = True
        
        with torch.no_grad():
            sum_check = attn_probs.sum(dim=-1)
            diff = torch.abs(sum_check - 1.0).max()
            if diff > 1e-4:
                print(f"Softmax norm check failed: max|sum-1| = {diff.item()}")

        attn_probs = self.attn_drop(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, length, n_channels)
        
        if not hasattr(self, '_logged_output_shape'):
            print(f"Attention output shape: {attn_output.shape}")
            self._logged_output_shape = True

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output
    
    def __str__(self):
        return f"MyAttention(dim={self.dim}, n_heads={self.n_heads}, bias={self.qkv.bias is not None})"
    
    def __repr__(self):
        return self.__str__()
        