import torch
import torch.nn as nn

from .attention import MyAttention
from .layers import MyMLP


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, emb_dim=192, patch_size=16, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    
    def __str__(self):
        return f"PatchEmbedding(in_channels={self.proj.in_channels}, emb_dim={self.proj.out_channels}, patch_size={self.patch_size}, img_size={self.img_size}, n_patches={self.n_patches})"
    
    def __repr__(self):
        return self.__str__()


class CLSToken(nn.Module):
    def __init__(self, emb_dim, num_cls=1):
        super(CLSToken, self).__init__()
        self.num_cls = num_cls
        self.cls_token = nn.Parameter(torch.zeros(1, num_cls, emb_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    
    def __str__(self):
        return f"CLSToken(emb_dim={self.cls_token.shape[-1]}, num_cls={self.num_cls})"
    
    def __repr__(self):
        return self.__str__()


class PositionalEmbedding(nn.Module):
    def __init__(self, width, height, emb_dim, num_cls=1):
        super(PositionalEmbedding, self).__init__()
        self.width = width
        self.height = height
        self.n_patches = width * height
        self.emb_dim = emb_dim
        self.num_cls = num_cls
        pos_emb = self._2d_positional_encoding(self.emb_dim)
        cls_emb = torch.zeros(1, num_cls, emb_dim)
        pos_emb = torch.cat([cls_emb, pos_emb.unsqueeze(0)], dim=1)
        self.register_buffer("pos_emb", pos_emb)
        
    def _1d_positional_encoding(self, length, dim):
        pos_emb = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb.unsqueeze(0)
    
    def _2d_positional_encoding(self, dim):
        emb_x = self._1d_positional_encoding(self.width, dim // 2).squeeze(0)
        emb_y = self._1d_positional_encoding(self.height, dim // 2).squeeze(0)
        emb_x = emb_x.unsqueeze(0).repeat(self.height, 1, 1)
        emb_y = emb_y.unsqueeze(1).repeat(1, self.width, 1)
        pos_embed = torch.cat([emb_x, emb_y], dim=-1)
        return pos_embed.reshape(self.n_patches, dim)

    def forward(self, x):
        batch_size, n_tokens, emb_dim = x.shape
        assert n_tokens == self.n_patches + self.num_cls
        pos_emb = self.pos_emb
        if pos_emb.device != x.device or pos_emb.dtype != x.dtype:
            pos_emb = pos_emb.to(device=x.device, dtype=x.dtype)
        x = x + pos_emb
        return x
    
    def __str__(self):
        return f"PositionalEmbedding(width={self.width}, height={self.height}, emb_dim={self.emb_dim}, n_patches={self.n_patches}, num_cls={self.num_cls})"
    
    def __repr__(self):
        return self.__str__()


class MiniViTEmbedding(nn.Module):
    def __init__(self, in_channels=3, emb_dim=192, patch_size=16, img_size=224, num_cls=1):
        super(MiniViTEmbedding, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, emb_dim, patch_size, img_size)
        n_patches_per_side = img_size // patch_size
        self.cls_token = CLSToken(emb_dim, num_cls)
        self.positional_embedding = PositionalEmbedding(n_patches_per_side, n_patches_per_side, emb_dim, num_cls)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.cls_token(x)
        x = self.positional_embedding(x)
        return x
    
    def __str__(self):
        return f"MiniViTEmbedding(patch_embedding={self.patch_embedding}, cls_token={self.cls_token}, positional_embedding={self.positional_embedding})"
    
    def __repr__(self):
        return self.__str__()
    

class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, mlp_hidden_dim, dropout=0.0, bias=False):
        super(EncoderBlock, self).__init__()

        self.attention = MyAttention(emb_dim, n_heads, bias=bias, dropout=dropout)
        self.mlp = MyMLP(emb_dim, mlp_hidden_dim, bias=bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
    def __str__(self):
        return f"EncoderBlock(emb_dim={self.norm1.normalized_shape[0]}, n_heads={self.attention.n_heads}, mlp_hidden_dim={self.mlp.fc1.out_features})"
    
    def __repr__(self):
        return self.__str__()
    
class Encoder(nn.Module):
    def __init__(self, n_layers, emb_dim, n_heads, mlp_hidden_dim, dropout=0.0, bias=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(emb_dim, n_heads, mlp_hidden_dim, dropout, bias)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
    
    def __str__(self):
        return f"MiniViTEncoder(n_layers={len(self.layers)}, emb_dim={self.norm.normalized_shape[0]})"
    
    def __repr__(self):
        return self.__str__()
    
class MiniViT(nn.Module):
    def __init__(self, in_channels=3, img_size=224, patch_size=16, emb_dim=192,
                 n_layers=6, n_heads=3, mlp_hidden_dim=768, dropout=0.1, bias=False,
                 n_classes=1000, num_cls=1):
        super(MiniViT, self).__init__()
        self.num_cls = num_cls
        self.emb_dim = emb_dim
        self.embedding = MiniViTEmbedding(in_channels, emb_dim, patch_size, img_size, num_cls)
        self.encoder = Encoder(n_layers, emb_dim, n_heads, mlp_hidden_dim, dropout, bias)
        self.classifier = nn.Linear(emb_dim * num_cls, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        cls_tokens = x[:, :self.num_cls, :]
        cls_tokens = cls_tokens.reshape(x.shape[0], self.num_cls * self.emb_dim)
        logits = self.classifier(cls_tokens)
        return logits
    
    def __str__(self):
        return f"MiniViT(embedding={self.embedding}, encoder={self.encoder}, classifier_out_features={self.classifier.out_features})"
    
    def __repr__(self):
        return self.__str__()
