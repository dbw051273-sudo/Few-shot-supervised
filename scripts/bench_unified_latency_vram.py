# -*- coding: utf-8 -*-
"""Unified single-GPU Batch=1 latency (ms/img) and peak VRAM (MB) benchmark for all models."""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPT_DIR.parent
THIRD_PARTY = PKG_ROOT / "third_party_official"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from round1_models import Codec, build_flow_model
from uni_encoder import UNIEncoder


def reset_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench(fn, warmup: int = 8, runs: int = 30) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0 / runs
    vram_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    return dt, vram_mb


def main() -> None:
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== Unified Single-GPU Benchmark (Batch=1, 256x256) on {gpu_name} ===")

    # 1. BaseFlow
    reset_vram()
    x = torch.randn(1, 3, 256, 256, device=dev)
    codec = Codec().to(dev).eval()
    base_flow = build_flow_model("u0_base", dim=192).to(dev).eval()
    def run_base():
        z0 = codec.encode(x)
        z = z0
        for k in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = base_flow(z, z0, z.new_full((1,), k * 0.25), None)
            z = z + v.float() * 0.25
        _ = codec.decode(z)
    ms, mem = bench(run_base)
    print(f"{'BaseFlow (4-step ODE)':34s} | {ms:7.2f} ms/img | Peak VRAM: {mem:7.1f} MB")
    del base_flow, codec, x

    # 2. UniFlow (4-step + frozen UNI)
    reset_vram()
    x = torch.randn(1, 3, 256, 256, device=dev)
    codec = Codec().to(dev).eval()
    uni_enc = UNIEncoder(freeze=True).to(dev).eval()
    uniflow = build_flow_model("u1_up64_only", dim=192).to(dev).eval()
    def run_uniflow():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ut = uni_enc(x)[:, 1:].float()
        z0 = codec.encode(x)
        z = z0
        for k in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = uniflow(z, z0, z.new_full((1,), k * 0.25), ut)
            z = z + v.float() * 0.25
        _ = codec.decode(z)
    ms, mem = bench(run_uniflow)
    print(f"{'UniFlow (4-step + UNI)':34s} | {ms:7.2f} ms/img | Peak VRAM: {mem:7.1f} MB")
    del uniflow, codec, uni_enc, x

    # 3. Pix2Pix & CycleGAN
    p2p_repo = THIRD_PARTY / "pytorch_CycleGAN_and_pix2pix"
    if p2p_repo.exists():
        sys.path.insert(0, str(p2p_repo))
        from models.networks import define_G
        reset_vram()
        x = torch.randn(1, 3, 256, 256, device=dev)
        net_p2p = define_G(3, 3, 64, "unet_256", "batch", True, "normal", 0.02).to(dev).eval()
        ms, mem = bench(lambda: net_p2p(x))
        print(f"{'Pix2Pix (UNet-256)':34s} | {ms:7.2f} ms/img | Peak VRAM: {mem:7.1f} MB")
        del net_p2p, x

        reset_vram()
        x = torch.randn(1, 3, 256, 256, device=dev)
        net_cyc = define_G(3, 3, 64, "resnet_9blocks", "instance", False, "normal", 0.02).to(dev).eval()
        ms, mem = bench(lambda: net_cyc(x))
        print(f"{'CycleGAN / CUT (ResNet-9)':34s} | {ms:7.2f} ms/img | Peak VRAM: {mem:7.1f} MB")
        del net_cyc, x
        sys.path.remove(str(p2p_repo))

    # 4. UNIStainNet
    uni_repo = THIRD_PARTY / "UNIStainNet"
    if uni_repo.exists():
        sys.path.insert(0, str(uni_repo))
        from src.models.generator import SPADEUNetGenerator
        reset_vram()
        x = torch.randn(1, 3, 256, 256, device=dev)
        uni_enc = UNIEncoder(freeze=True).to(dev).eval()
        gen = SPADEUNetGenerator(
            num_classes=5, class_dim=64, uni_dim=1024, input_skip=True,
            edge_encoder="v2", edge_base_ch=32, uni_spatial_size=32, image_size=512,
        ).to(dev).eval()
        labels_in = torch.zeros(1, dtype=torch.long, device=dev)
        def run_unistainnet():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok14 = uni_enc(x)[:, 1:, :].transpose(1, 2).reshape(1, 1024, 14, 14)
                tok32 = F.interpolate(tok14, size=32, mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
                x512 = F.interpolate(x, size=512, mode="bilinear", align_corners=False)
                out512 = gen(x512, tok32, labels_in)
                _ = F.interpolate(out512, size=256, mode="bilinear", align_corners=False)
        ms, mem = bench(run_unistainnet)
        print(f"{'UNIStainNet (SPADE-UNet + UNI)':34s} | {ms:7.2f} ms/img | Peak VRAM: {mem:7.1f} MB")
        del gen, uni_enc, labels_in, x
        sys.path.remove(str(uni_repo))

    # 5. I2SB (20-step ADM UNet)
    i2sb_repo = THIRD_PARTY / "I2SB"
    if i2sb_repo.exists():
        sys.path.insert(0, str(i2sb_repo))
        from guided_diffusion.script_util import create_model
        reset_vram()
        x = torch.randn(1, 3, 256, 256, device=dev)
        unet = create_model(
            image_size=256, num_channels=128, num_res_blocks=2, channel_mult="",
            learn_sigma=False, class_cond=False, use_checkpoint=False, attention_resolutions="32,16,8",
            num_heads=4, num_head_channels=64, num_heads_upsample=-1, use_scale_shift_norm=True,
            dropout=0.0, resblock_updown=True, use_fp16=False, use_new_attention_order=False,
            in_channels=6,
        ).to(dev).eval()
        t_zero = torch.zeros(1, device=dev)
        def run_i2sb():
            cur = x
            for _ in range(20):
                _ = unet(torch.cat([cur, x], dim=1), t_zero)
        ms, mem = bench(run_i2sb, warmup=2, runs=8)
        print(f"{'I2SB (20-step ADM UNet)':34s} | {ms:7.2f} ms/img | Peak VRAM: {mem:7.1f} MB")
        del unet, x
        sys.path.remove(str(i2sb_repo))


if __name__ == "__main__":
    main()
