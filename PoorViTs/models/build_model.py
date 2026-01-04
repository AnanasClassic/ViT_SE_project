import gin.config
from .vit import VisionTransformer
from .swin import SwinTransformer
from .cait import cait_models
from functools import partial
from torch import nn
from pathlib import Path
import sys

import gin
import torch


def create_model(img_size, n_classes, args):

    if args.arch == "vit":
        patch_size = 4 if img_size == 32 else 8   #4 if img_size = 32 else 8
        model = VisionTransformer(img_size=[img_size],
            patch_size=args.patch_size,
            in_chans=3,
            num_classes=n_classes,
            # embed_dim=192,
            # depth=9,
            # num_heads=12,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=True,
            drop_path_rate=args.sd,
            sin_pos=args.sin_pos,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # import timm.models.vision_transformer as timm_vit
        # model = VisionTransformer(
        #     img_size=img_size,
        #     patch_size=[patch_size,patch_size],
        #     in_chans=3,
        #     num_classes=n_classes,
        #     embed_dim=192,
        #     depth=9,
        #     num_heads=12,
        #     mlp_ratio=args.vit_mlp_ratio,
        #     qkv_bias=True,
        #     drop_rate=args.sd,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # )

    elif args.arch == 'cait':       
        patch_size = 4 if img_size == 32 else 8
        model = cait_models(
        img_size= img_size,patch_size=patch_size, embed_dim=192, depth=24, num_heads=4, mlp_ratio=args.vit_mlp_ratio,
        qkv_bias=True,num_classes=n_classes,drop_path_rate=args.sd,norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,depth_token_only=2)
    
        
    elif args.arch =='swin':
        
        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if img_size==32 else 4

        model = SwinTransformer(img_size=img_size,
        window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],num_classes=n_classes,
       	mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.sd)
    elif args.arch == 'mini_vit':
        try:
            from src.models.mini_vit import MiniViT
        except ModuleNotFoundError:
            project_root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(project_root))
            from src.models.mini_vit import MiniViT
        emb_dim = args.channel if args.channel is not None else 192
        n_layers = args.depth if args.depth is not None else 12
        n_heads = args.heads if args.heads is not None else 12
        mlp_hidden_dim = emb_dim * args.vit_mlp_ratio
        model = MiniViT(
            in_channels=3,
            img_size=img_size,
            patch_size=args.patch_size,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=0.0,
            bias=True,
            n_classes=n_classes,
        )
    elif args.arch == 'mhc_vit':
        try:
            from src.models.mhc_mini_vit import MHC_MiniViT
        except ModuleNotFoundError:
            project_root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(project_root))
            from src.models.mhc_mini_vit import MHC_MiniViT
        emb_dim = args.channel if args.channel is not None else 192
        n_layers = args.depth if args.depth is not None else 12
        n_heads = args.heads if args.heads is not None else 12
        mlp_hidden_dim = emb_dim * args.vit_mlp_ratio
        model = MHC_MiniViT(
            in_channels=3,
            img_size=img_size,
            patch_size=args.patch_size,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=0.0,
            bias=True,
            n_classes=n_classes,
        )
    elif args.arch == 'none':
        class NoneModel(nn.Module):
            def __init__(self):
                super(NoneModel, self).__init__()
                self.model = nn.Parameter(torch.randn(1, 10,requires_grad=True))
            def forward(self, x):
                
                return self.model.repeat(x.shape[0],1)
        model = NoneModel()
    else:
        NotImplementedError("Model architecture not implemented . . .")

         
    return model
