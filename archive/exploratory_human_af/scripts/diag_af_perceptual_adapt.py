# -*- coding: utf-8 -*-
"""Quick 3-min Phase 2B Pilot diagnostic: test perceptual-weighted SRA (lower L1, higher LPIPS)
on already-pretrained Human Breast AF checkpoints (raw & p1p99) without changing architecture.
"""
import copy, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from cellpose import models as cp_models
from PIL import Image

SCRIPT_DIR = Path(r"C:\Users\felix\Desktop\Few-shot supervised（小样本有监督）\10_GitHub开源仓库_Few-shot-supervised\scripts")
sys.path.insert(0, str(SCRIPT_DIR))
from round1_models import Codec, SwadAccumulator, build_flow_model
from round1_losses_and_data import augment_pair, grad_l1, infer_to_pngs, pi_acq_jitter, stain_aug_source_he
from uni_encoder import UNIEncoder
from v2_unified_eval import build_per_image_lpips, evaluate_split, load_gt_instance_masks
from run_af_human_pipeline import DEFAULT_AF_DATA, DEFAULT_AF_NPZ, DEFAULT_AF_WORK, load_af_fold_tensors


def adapt_perceptual_af(
    tag, flow, swad_st, init_codec, center, scale, src_pools, tgt_pi, tgt_he,
    lpips_fn, uni_enc, w_l1=1.0, w_grad=1.0, w_lpips=2.0, steps=1500, batch=4, seed=7202,
):
    torch.manual_seed(seed)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    py_rng = random.Random(seed)
    flow.load_state_dict(swad_st)
    flow.cuda().train()
    codec = copy.deepcopy(init_codec).cuda()
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    codec.encoder.eval()
    codec.decoder.train()
    for p in codec.decoder.parameters():
        p.requires_grad_(True)
    params = list(flow.parameters()) + list(codec.decoder.parameters())
    opt = torch.optim.AdamW([{"params": list(flow.parameters()), "lr": 2e-4}, {"params": list(codec.decoder.parameters()), "lr": 2e-5}], weight_decay=1e-4)
    swad = SwadAccumulator({"flow": flow, "decoder": codec.decoder}, start_step=600, every=10)
    t_pi, t_he = tgt_pi.cuda(), tgt_he.cuda()
    for step in range(1, steps + 1):
        ti = torch.randint(len(t_pi), (batch,), generator=rng, device="cuda")
        xt, yt = augment_pair(t_pi[ti], t_he[ti], py_rng)
        xt_in = pi_acq_jitter(xt, py_rng, strength=0.05)
        with torch.no_grad():
            z0_t = (codec.encode(xt_in).float() - center) / scale
            z1_t = (codec.encode(yt).float() - center) / scale
            ut_t = uni_enc(xt_in)[:, 1:].float() if (uni_enc is not None and getattr(flow, "use_uni_2d", False)) else None
        tt = torch.rand(batch, generator=rng, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_t = flow(torch.lerp(z0_t, z1_t, tt[:, None, None, None]), z0_t, tt, ut_t)
            z = z0_t
            for k in range(4):
                z = z + flow(z, z0_t, z.new_full((batch,), k * 0.25), ut_t) * 0.25
            pred_t = codec.decode(z * scale + center)
        l_vel = F.mse_loss(v_t.float(), z1_t - z0_t)
        l_img = w_l1 * F.l1_loss(pred_t.float(), yt) + w_grad * grad_l1(pred_t.float(), yt)
        l_lp = lpips_fn((pred_t.float().clamp(-1, 1) + 1.0) * 0.5, (yt.clamp(-1, 1) + 1.0) * 0.5).mean()
        loss = l_vel + l_img + w_lpips * l_lp

        op_pi, op_he = src_pools[py_rng.randrange(len(src_pools))]
        si = torch.randint(len(op_pi), (batch,), generator=rng, device="cuda")
        xs, ys = augment_pair(op_pi[si], op_he[si], py_rng)
        xs_in = pi_acq_jitter(xs, py_rng, strength=0.08)
        ys_in = stain_aug_source_he(ys, t_he, py_rng)
        with torch.no_grad():
            z0_s = (codec.encode(xs_in).float() - center) / scale
            z1_s = (codec.encode(ys_in).float() - center) / scale
            ut_s = uni_enc(xs_in)[:, 1:].float() if (uni_enc is not None and getattr(flow, "use_uni_2d", False)) else None
        ts = torch.rand(batch, generator=rng, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_s = flow(torch.lerp(z0_s, z1_s, ts[:, None, None, None]), z0_s, ts, ut_s)
            rec_s = codec.decode(z1_s * scale + center)
        l_s = 0.6 * F.mse_loss(v_s.float(), z1_s - z0_s) + 0.6 * F.l1_loss(rec_s.float(), ys_in)
        loss = loss + 0.40 * l_s

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.5)
        opt.step()
        swad.maybe_update(step, steps)
    swad.apply()
    return flow.eval().cpu(), codec.eval().cpu()


def main():
    device = torch.device("cuda:0")
    splits = json.loads((DEFAULT_AF_DATA / "af_lomo_splits.json").read_text(encoding="utf-8"))
    uni_enc = UNIEncoder(freeze=True).cuda().eval()
    lpips_fn = build_per_image_lpips(net_type="alex", normalize=True).to(device)
    cp_model = cp_models.CellposeModel(gpu=True, model_type="nuclei")
    gt_masks = load_gt_instance_masks(DEFAULT_AF_NPZ, "Breast")

    for af_mode in ("raw", "p1p99"):
        fd = load_af_fold_tensors(DEFAULT_AF_DATA, splits, "Breast", af_mode, k_shot=20)
        ckpt_dir = DEFAULT_AF_WORK / af_mode / "checkpoints" / "Breast"
        codec = Codec().cuda()
        codec.load_state_dict(torch.load(ckpt_dir / "codec_rae0.08_s8800.pth", map_location="cuda", weights_only=True))
        codec.eval().requires_grad_(False)
        organ_pools = [(fd["src_pi"][[i for i, g in enumerate(fd["src_groups"]) if g == o]].cuda(), fd["src_he"][[i for i, g in enumerate(fd["src_groups"]) if g == o]].cuda()) for o in sorted(set(fd["src_groups"]))]

        for arch in ("u0_base", "u1_up64_only"):
            pre = torch.load(ckpt_dir / f"Breast__{arch}_d192_pretrain.pth", map_location="cpu", weights_only=False)
            fl_ad, co_ad = adapt_perceptual_af(
                f"{af_mode}_{arch}_perc", build_flow_model(arch, dim=192), pre["swad"], codec,
                pre["center"], pre["scale"], organ_pools, fd["adapt_pi_t"], fd["adapt_he_t"], lpips_fn, uni_enc,
                w_l1=1.0, w_grad=1.0, w_lpips=2.0, steps=1500,
            )
            out_dir = DEFAULT_AF_WORK / af_mode / "generated" / "Breast" / f"Breast__{arch}_d192__m6_k20_perc"
            pmap = infer_to_pngs(fl_ad, co_ad, fd["test_pi_t"], uni_enc, pre["center"], pre["scale"], fd["test_ids"], out_dir, steps=4)
            m = evaluate_split(fd["test_ids"], pmap, fd["pi_rgb_map"], fd["gt_rgb_map"], gt_masks, cp_model, lpips_fn, device)
            print(f"[PERC PILOT] {af_mode:5s} {arch:12s} | PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} KID={m['kid']*1000:.2f} Dice={m['nuclei_dice']:.4f} Dice_ne={m['nuclei_dice_nonempty']:.4f} F1={m['nuclei_f1_05']:.4f}", flush=True)


if __name__ == "__main__":
    main()
