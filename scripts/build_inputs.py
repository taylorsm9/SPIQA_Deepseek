"""Build and freeze the per-arm input messages for a sample set.

  python scripts/build_inputs.py --arms A,B,C,D,O --set questions
  python scripts/build_inputs.py --arms N --set arm_n
  python scripts/build_inputs.py --arms D,Dnoimg --set vision_check

Inputs are written once; an existing input is never overwritten (pass --check to verify hashes instead).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.prompts import build
from intake.spiqa import load_all, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--set", default="questions")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare hashes")
    args = ap.parse_args()
    cfg = load_config()
    papers = load_all(cfg)
    qmap = {q.qid: (q, p) for p in papers.values() for q in p.questions}
    qids = json.load(open(cfg["paths"]["samples"] / f"stage{args.stage}.json"))[args.set]

    for arm in args.arms.split(","):
        out_dir = cfg["paths"]["inputs"] / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        stats = Counter()
        for qid in qids:
            path = out_dir / (qid.replace("/", "__") + ".json")
            q, p = qmap[qid]
            if path.exists() and not args.check:
                stats["existing"] += 1
                continue
            inp = build(arm, q, p, cfg)
            if args.check:
                same = json.load(open(path))["input_sha256"] == inp["input_sha256"]
                stats["match" if same else "MISMATCH"] += 1
                continue
            json.dump(inp, open(path, "w"))
            stats["built"] += 1
            stats["images"] += inp["n_images"]
            stats["with_dropped_images"] += bool(inp.get("images_dropped"))
            if "refs" in inp:
                stats["refs"] += inp["refs"]
                stats["refs_matched"] += inp["refs_matched_in_ocr"]
        print(arm, dict(stats))


if __name__ == "__main__":
    main()
