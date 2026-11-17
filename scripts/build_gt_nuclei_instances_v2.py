"""Extract and cache uint16 Cellpose nuclei instance masks for all 4 organs' test GT H&E in UniPrior_v2."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from cellpose import models
from PIL import Image


def load_rgb_resized(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def main():
    default_data = Path(__file__).resolve().parents[2] / 'UniPrior_v2'
    default_cache = Path(__file__).resolve().parents[1] / 'eval_cache'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=default_data)
    parser.add_argument('--out-dir', type=Path, default=default_cache)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f'gt_nuclei_instances_{args.size}.npz'

    cm = models.CellposeModel(gpu=True, model_type='nuclei')
    keys, masks_list, summary = [], [], {}

    for organ in ('liver', 'Kidney', 'Brain', 'Breast'):
        t_dir = data_dir / organ / 'test' / 'target'
        files = sorted(t_dir.glob('*.png'))
        imgs = [load_rgb_resized(p, args.size) for p in files]
        organ_masks = []
        for i in range(0, len(imgs), args.batch_size):
            mb, _, _ = cm.eval(imgs[i:i + args.batch_size], diameter=None, channels=[0, 0], batch_size=len(imgs[i:i + args.batch_size]))
            organ_masks.extend(m.astype(np.uint16) for m in mb)
        inst_counts = [int(m.max()) for m in organ_masks]
        fg_pcts = [float((m > 0).mean() * 100.0) for m in organ_masks]
        summary[organ] = {
            'count': len(files),
            'median_nuclei': float(np.median(inst_counts)),
            'mean_nuclei': round(float(np.mean(inst_counts)), 2),
            'mean_fg_pct': round(float(np.mean(fg_pcts)), 2),
            'zero_mask_images': int(sum(1 for c in inst_counts if c == 0)),
        }
        print(f"[GT_INST] {organ:8s}: n={len(files)} median_nuclei={summary[organ]['median_nuclei']:.1f}", flush=True)
        for p, m in zip(files, organ_masks):
            keys.append(f'{organ}/{p.name}')
            masks_list.append(m)

    np.savez_compressed(out_npz, keys=np.array(keys), masks=np.stack(masks_list, axis=0))
    meta = {'size': args.size, 'total_masks': len(keys), 'organs': summary, 'elapsed_s': round(time.perf_counter() - t0, 2)}
    (out_dir / f'gt_nuclei_instances_{args.size}_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Saved {out_npz} ({out_npz.stat().st_size / 1024:.1f} KB) in {meta["elapsed_s"]}s', flush=True)


if __name__ == '__main__':
    main()
