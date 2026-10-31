"""Draw the stratified paper sample and its subsets (no-paper questions, pilot, dry run, vision check).

Stratified by split, and within split by whether the paper's references lean on figures or tables,
allocated proportionally.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.spiqa import load_all, load_config


def allocate(n: int, sizes: dict[str, int]) -> dict[str, int]:
    """Largest-remainder proportional allocation of n across strata."""
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(r) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def draw(papers, n_per_split, rng, exclude=frozenset(), keep=frozenset()):
    """Stratified draw. Papers in `keep` stay in and count toward their stratum's allocation."""
    chosen = []
    for split, n in sorted(n_per_split.items()):
        strata = {}
        for key, p in sorted(papers.items()):
            if p.split == split and key not in exclude:
                strata.setdefault(p.lean, []).append(key)
        alloc = allocate(n, {k: len(v) for k, v in strata.items()})
        for lean in sorted(strata):
            kept = [k for k in strata[lean] if k in keep]
            rest = [k for k in strata[lean] if k not in keep]
            chosen += kept + rng.sample(rest, max(0, alloc[lean] - len(kept)))
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()

    cfg = load_config()
    sc = cfg["sample"]
    papers = load_all(cfg)
    out_dir = cfg["paths"]["samples"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # RNG derived from the recorded seed
    rng = random.Random(f"{sc['seed']}-stage{args.stage}")

    exclude = set()
    if args.stage == 2:
        exclude = set(json.load(open(out_dir / "stage1.json"))["papers"])
    prev_path = out_dir / f"stage{args.stage}.json"
    prev = json.load(open(prev_path)) if prev_path.exists() else None
    if prev:
        # Extend an earlier, smaller draw (100 papers to 300). Its papers stay in and its subsets are kept.
        rng = random.Random(f"{sc['seed']}-stage{args.stage}-extend{sum(sc[f'stage{args.stage}'].values())}")
        json.dump(prev, open(out_dir / f"stage{args.stage}_initial.json", "w"), indent=1)
    keys = draw(papers, sc[f"stage{args.stage}"], rng, exclude, keep=set(prev["papers"]) if prev else set())
    questions = [q.qid for k in keys for q in papers[k].questions]
    result = {"seed": sc["seed"], "stage": args.stage, "papers": keys, "questions": questions}

    if prev:
        for k in ("arm_n", "flip_pilot", "vision_check", "dry_run"):
            if k in prev:
                result[k] = prev[k]
        result["initial_papers"] = prev["papers"]
    elif args.stage == 1:
        sub = random.Random(f"{sc['seed']}-subsets")
        result["arm_n"] = sorted(sub.sample(questions, sc["arm_n_questions"]))
        result["flip_pilot"] = sorted(sub.sample(questions, sc["flip_pilot_questions"]))
        # Vision check: test-A plot questions from papers outside the sample (no OCR needed, arm D inputs)
        plot_qs = [q.qid for k, p in sorted(papers.items()) if p.split == "A" and k not in keys
                   for q in p.questions if q.ref_types == ["plot"]]
        result["vision_check"] = sorted(sub.sample(plot_qs, sc["vision_check_questions"]))
        result["dry_run"] = sorted(sub.sample(questions, sc["dry_run_per_arm"]))

    with open(out_dir / f"stage{args.stage}.json", "w") as f:
        json.dump(result, f, indent=1)

    for split in "ABC":
        ks = [k for k in keys if papers[k].split == split]
        nq = sum(len(papers[k].questions) for k in ks)
        nt = sum(papers[k].lean == "table" for k in ks)
        print(f"test-{split}: {len(ks)} papers ({nt} table-leaning), {nq} questions")
    print(f"total: {len(keys)} papers, {len(questions)} questions -> {out_dir / f'stage{args.stage}.json'}")


if __name__ == "__main__":
    main()
