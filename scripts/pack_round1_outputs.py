"""Pack Round-1 outputs (metrics, CSVs, visuals, and optionally generated images/checkpoints) into exports/."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import zipfile


def build_archive(
    work_dir: Path,
    out_zip: Path,
    include_generated: bool = True,
    include_checkpoints: bool = False,
) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    roots = ['eval_results', 'visuals', 'scripts', 'third_party_official']
    if include_generated:
        roots.append('generated')
    if include_checkpoints:
        roots.append('checkpoints')

    added = 0
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_root in roots:
            p_root = work_dir / rel_root
            if not p_root.exists():
                continue
            for fp in sorted(p_root.rglob('*')):
                if fp.is_file() and '__pycache__' not in fp.parts and not any(pt.startswith('_tmp_refs_') for pt in fp.parts):
                    zf.write(fp, fp.relative_to(work_dir).as_posix())
                    added += 1
    print(f'[PACK] Wrote {added} files -> {out_zip} ({out_zip.stat().st_size / (1024 * 1024):.2f} MB)', flush=True)
    return out_zip


def main() -> None:
    default_work = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', type=Path, default=default_work)
    ap.add_argument('--tag', type=str, default='round1')
    ap.add_argument('--include-generated', action='store_true', default=True)
    ap.add_argument('--no-generated', dest='include_generated', action='store_false')
    ap.add_argument('--include-checkpoints', action='store_true', default=False)
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    lite_zip = work_dir / 'exports' / f'{args.tag}_metrics_visuals_{ts}.zip'
    build_archive(work_dir, lite_zip, include_generated=False, include_checkpoints=False)
    if args.include_generated:
        full_zip = work_dir / 'exports' / f'{args.tag}_with_images_{ts}.zip'
        build_archive(work_dir, full_zip, include_generated=True, include_checkpoints=args.include_checkpoints)


if __name__ == '__main__':
    main()
