"""Freeze MinerU output: unpack each zip, build reading-order blocks, cut crops, hash everything.

Run after mineru-kit has parsed data/pdfs into data/mineru/zips. Writes data/mineru/frozen/<pdf stem>/
with blocks.json, crops/ and the raw MinerU files, plus data/mineru/frozen/manifest.json (sha256 of every
file) so later arms can prove they read the same OCR.

  python scripts/freeze_mineru.py                          # standard tier -> data/mineru (arms A, B, C, O)
  python scripts/freeze_mineru.py --out mineru_adv_out     # advanced tier -> data/mineru_adv (arm E), no crops
"""

import argparse

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.mineru import parse_middle, render_crops, save_blocks
from intake.spiqa import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mineru_out", help="config [paths] key of the MinerU output folder")
    args = ap.parse_args()
    cfg = load_config()
    zips = cfg["paths"][args.out] / "zips"
    frozen = cfg["paths"][args.out] / "frozen"
    crops = args.out == "mineru_out"   # arm E reads text only
    pdfs = cfg["paths"]["pdfs"]
    manifest = {}
    stats = []
    for z in sorted(zips.glob("*.zip")):
        out = frozen / z.stem
        if not (out / "blocks.json").exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out / "raw")
            blocks = parse_middle(json.load(open(out / "raw/middle_json.json")))
            if crops:
                render_crops(pdfs / f"{z.stem}.pdf", blocks, out / "crops")
            save_blocks(out / "blocks.json", blocks)
        blocks = json.load(open(out / "blocks.json"))
        n_fig = sum(b["kind"] == "figure" for b in blocks)
        n_tab = sum(b["kind"] == "table" for b in blocks)
        n_nocap = sum(b["kind"] in ("figure", "table") and not b["caption"] for b in blocks)
        n_desc = sum(b["kind"] == "figure" and bool(b.get("desc")) for b in blocks)
        stats.append((z.stem, n_fig, n_tab, n_nocap, n_desc))
        for f in sorted(out.rglob("*")):
            if f.is_file():
                manifest[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    json.dump(manifest, open(frozen / "manifest.json", "w"), indent=0)
    for s in stats:
        print("%-28s figures=%-3d tables=%-3d uncaptioned=%-3d described=%d" % s)
    print(f"{len(stats)} papers frozen, {len(manifest)} files hashed")


if __name__ == "__main__":
    main()
