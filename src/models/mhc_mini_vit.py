import torch
import torch.nn as nn
from .mini_vit import MiniViTEmbedding, EncoderBlock

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class DoublyStochasticLayer(torch.nn.Module):
    def __init__(self, iters: int=20, eps: float=1e-8):
        super(DoublyStochasticLayer, self).__init__()
        self.iters = iters
        self.eps = eps

    def _axis_normalize(self, matrix: torch.Tensor, axis: int):
        return matrix / (torch.sum(matrix, dim=axis, keepdim=True) + self.eps)

    def forward(self, matrix: torch.Tensor):
        matrix = matrix - torch.amax(matrix, dim=(-2, -1), keepdim=True)
        matrix = torch.exp(matrix)
        for iter in range(self.iters):
            matrix = self._axis_normalize(matrix, -1)
            matrix = self._axis_normalize(matrix, -2)
        return matrix

    def __str__(self):
        return f'DoublyStochasticLayer(iters={self.iters})'

    def __repr__(self):
        return self.__str__()


class MHCEncoderBlock(nn.Module):
    def __init__(self, n_layers=4, emb_dim=192, n_heads=3, mlp_hidden_dim=768, n_streams=4,
                 dropout=0.1, bias=False, ds_iters=20):
        super(MHCEncoderBlock, self).__init__()

        self.n_layers = n_layers
        self.n_streams = n_streams
        self.emb_dim = emb_dim

        self.norm = RMSNorm(emb_dim * n_streams)
        self.ds = DoublyStochasticLayer(iters=ds_iters)

        self.encoder = EncoderBlock(emb_dim, n_heads, mlp_hidden_dim, dropout, bias)

        self.phi_pre = nn.Parameter(torch.empty(emb_dim * n_streams, n_streams))
        self.phi_post = nn.Parameter(torch.empty(emb_dim * n_streams, n_streams))
        self.phi_res = nn.Parameter(torch.empty(emb_dim * n_streams, n_streams * n_streams))

        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams, n_streams))

        self.alpha_pre = nn.Parameter(torch.tensor(1e-3))
        self.alpha_post = nn.Parameter(torch.tensor(1e-3))
        self.alpha_res = nn.Parameter(torch.tensor(1e-3))

        nn.init.normal_(self.phi_pre, std=0.02)
        nn.init.normal_(self.phi_post, std=0.02)
        nn.init.normal_(self.phi_res, std=0.02)

        with torch.no_grad():
            self.b_pre.fill_(float(torch.log(torch.tensor(1.0 / (n_streams - 1.0)))))
            self.b_post.zero_()
            self.b_res.fill_(-10.0)
            self.b_res.diagonal().fill_(0.0)

    def forward(self, x):
        b, t, n, d = x.shape

        streams = x.reshape(b, t, n * d)
        streams = self.norm(streams)

        pre_tilde = self.alpha_pre * (streams @ self.phi_pre) + self.b_pre
        post_tilde = self.alpha_post * (streams @ self.phi_post) + self.b_post
        res_tilde = self.alpha_res * (streams @ self.phi_res) + self.b_res.reshape(1, 1, n * n)

        h_pre = torch.sigmoid(pre_tilde)
        h_post = 2.0 * torch.sigmoid(post_tilde)
        h_res = self.ds(res_tilde.reshape(b, t, n, n))

        x_res = torch.einsum('btij,btjd->btid', h_res, x)
        x_in = torch.einsum('btn,btnd->btd', h_pre, x)
        y = self.encoder(x_in)
        x_out = y.unsqueeze(-2) * h_post.unsqueeze(-1)

        return x_res + x_out

    def __str__(self):
        return f'MHCEncoderBlock(n_layers={self.n_layers}, emb_dim={self.emb_dim}, n_streams={self.n_streams})'

    def __repr__(self):
        return self.__str__()


class MHC_MiniViT(nn.Module):
    def __init__(self, in_channels=3, img_size=224, patch_size=16, emb_dim=192,
                 n_layers=6, n_heads=3, mlp_hidden_dim=768, n_streams=4,
                 dropout=0.1, bias=False, ds_iters=20,
                 n_classes=1000):
        super(MHC_MiniViT, self).__init__()
        self.n_streams = n_streams
        self.embedding = MiniViTEmbedding(in_channels, emb_dim, patch_size, img_size)
        self.encoder = nn.ModuleList([
            MHCEncoderBlock(n_layers, emb_dim, n_heads, mlp_hidden_dim, n_streams, dropout, bias, ds_iters)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        b, t, d = x.shape
        x = x.unsqueeze(-2).expand(b, t, self.n_streams, d)
        for layer in self.encoder:
            x = layer(x)
        x = x[:, 0].mean(dim=-2)
        x = self.classifier(x)
        return x

    def __str__(self):
        return f'MHC_MiniViT(embedding={self.embedding}, encoder={self.encoder}, classifier={self.classifier})'

    def __repr__(self):
        return self.__str__()
