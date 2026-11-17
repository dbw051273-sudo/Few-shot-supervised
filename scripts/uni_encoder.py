# -*- coding: utf-8 -*-
"""Self-contained frozen UNI pathological foundation model wrapper."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class UNIEncoder(nn.Module):
    """Frozen MahmoodLab/UNI encoder extracting 14x14 patch + CLS tokens."""

    def __init__(self, model_name: str = "hf-hub:MahmoodLab/UNI", freeze: bool = True):
        super().__init__()
        self.uni = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.embed_dim = self.uni.embed_dim

        if freeze:
            for p in self.uni.parameters():
                p.requires_grad = False
            self.uni.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() < -0.5:
            x = (x + 1.0) / 2.0
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_cls: bool = False) -> torch.Tensor:
        was_training = self.uni.training
        if was_training:
            self.uni.eval()
        tokens = self.uni.forward_features(self._preprocess(x))
        out = tokens[:, 0] if return_cls else tokens
        if was_training:
            self.uni.train()
        return out
