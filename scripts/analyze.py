"""Every result in the README, computed from the judged runs.

  python scripts/analyze.py --runs stage1,stage1_flash
  python scripts/analyze.py --runs phase0_vision --vision   # check that images reach the model

Scores are averaged over seeds per question (one seed here), then compared question by question.
Every average weights the three splits equally. Intervals resample papers within each split, not
questions, because questions from one paper are correlated.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.spiqa import ROOT, load_all, load_config

# SPIQA's reference categories; intake/spiqa.py normalizes "photograph(s)" to "photo" (SPIQA's visualizations)
REF_GROUPS = {"table": "table", "plot": "plot", "schematic": "schematic", "photo": "visualization",
              "other": "other figure"}


def load_jsonl_last(path):
    """Last record per id, so a retried generation replaces its failed attempt."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "error" not in r or r["id"] not in out:
            out[r["id"]] = r
    return out


def load_scores(cfg, runs):
    """{arm: {qid: mean L3 over seeds, in points}}"""
    per = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for r in load_jsonl_last(cfg["paths"]["runs"] / run / "scores.jsonl").values():
            per[r["arm"]][r["qid"]].append(100 * r["l3"])
    return {a: {q: float(np.mean(v)) for q, v in qs.items()} for a, qs in per.items()}


class Frame:
    """Questions shared by the arms being compared, grouped by split and paper for the bootstrap."""

    def __init__(self, scores, arms, qinfo, only=None):
        common = set.intersection(*(set(scores[a]) for a in arms))
        if only is not None:
            common &= set(only)
        self.qids = sorted(common)
        self.split = np.array([qinfo[q]["split"] for q in self.qids])
        self.paper = np.array([qinfo[q]["paper"] for q in self.qids])
        self.ref = np.array([qinfo[q]["ref_group"] for q in self.qids])
        self.s = {a: np.array([scores[a][q] for q in self.qids]) for a in arms}
        self.groups = {sp: {} for sp in "ABC"}
        for i, (sp, p) in enumerate(zip(self.split, self.paper)):
            self.groups[sp].setdefault(p, []).append(i)

    def weighted(self, x, idx_by_split):
        means = [x[idx].mean() for idx in idx_by_split.values() if len(idx)]
        return float(np.mean(means)) if means else float("nan")

    def full_idx(self, mask=None):
        out = {}
        for sp in "ABC":
            idx = np.array([i for ids in self.groups[sp].values() for i in ids], dtype=int)
            if mask is not None:
                idx = idx[mask[idx]]
            out[sp] = idx
        return out

    def boot(self, fn, reps, seed=0, mask=None):
        rng = np.random.default_rng(seed)
        papers = {sp: list(self.groups[sp].values()) for sp in "ABC"}
        out = np.empty(reps)
        for r in range(reps):
            idx = {}
            for sp, plist in papers.items():
                if not plist:
                    continue
                pick = rng.integers(0, len(plist), len(plist))
                ids = np.array([i for k in pick for i in plist[k]], dtype=int)
                idx[sp] = ids[mask[ids]] if mask is not None else ids
            out[r] = fn(idx)
        return out


def compare(fr, a, b, level, reps, mask=None):
    """Mean of each arm and the difference b - a with its paper-bootstrap interval."""
    idx = fr.full_idx(mask)
    d = fr.s[b] - fr.s[a]
    fn = lambda ix: fr.weighted(d, ix)
    bs = fr.boot(fn, reps, mask=mask)
    lo, hi = np.nanpercentile(bs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {a: round(fr.weighted(fr.s[a], idx), 2), b: round(fr.weighted(fr.s[b], idx), 2),
            f"{b}-{a}": round(fn(idx), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2),
            "n": int(sum(len(v) for v in idx.values()))}


def by_split_and_overall(fr, a, b, level, reps):
    out = {"overall": compare(fr, a, b, level, reps)}
    for sp in "ABC":
        out[f"test-{sp}"] = compare(fr, a, b, level, reps, mask=fr.split == sp)
    return out


def reliability_and_cost(cfg, runs):
    gens, ans_tok = {}, {}
    for run in runs:
        run_dir = cfg["paths"]["runs"] / run
        gens.update(load_jsonl_last(run_dir / "generations.jsonl"))
        if (run_dir / "answer_tokens.jsonl").exists():
            ans_tok.update({r["id"]: r["answer_tokens"] for r in load_jsonl_last(run_dir / "answer_tokens.jsonl").values()})
    by_arm = defaultdict(list)
    for g in gens.values():
        by_arm[g["arm"]].append(g)

    def pct(v):
        v = [x for x in v if x is not None]
        return {"median": round(float(np.median(v)), 1), "p90": round(float(np.percentile(v, 90)), 1),
                "n": len(v)} if v else None

    rel, cost = {}, {}
    for arm, gs in sorted(by_arm.items()):
        n = len(gs)
        ok = [g for g in gs if "error" not in g]
        rel[arm] = {"n": n, "truncated": round(sum(bool(g.get("truncated")) for g in ok) / n, 4),
                    "errors": round((n - len(ok)) / n, 4),
                    "unparsed": round(sum(not g.get("parsed") for g in ok) / n, 4)}
        reasoning = [g["completion_tokens"] - ans_tok[g["id"]] for g in ok
                     if g.get("completion_tokens") is not None and g["id"] in ans_tok]
        cost[arm] = {"output_tokens": pct([g.get("completion_tokens") for g in ok]),
                     "reasoning_tokens": pct(reasoning),
                     "answer_tokens": pct([ans_tok.get(g["id"]) for g in ok]),
                     "latency_s": pct([g.get("latency_s") for g in ok])}
    return rel, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = load_config()
    ac = cfg["analysis"]
    level, reps = ac["ci"], ac["bootstrap_reps"]
    runs = args.runs.split(",")
    papers = load_all(cfg)
    qinfo = {}
    for p in papers.values():
        for q in p.questions:
            types = set(q.ref_types)
            group = REF_GROUPS[types.pop()] if len(types) == 1 else "multiple types"
            qinfo[q.qid] = {"split": p.split, "paper": p.key, "ref_group": group}
    scores = load_scores(cfg, runs)

    if args.vision:
        fr = Frame(scores, ["Dnoimg", "D"], qinfo)
        gain = float(np.mean(fr.s["D"] - fr.s["Dnoimg"]))
        # any gain shows the images reach the model
        report = {"vision_check": {"n": len(fr.qids), "with_images": float(fr.s["D"].mean()),
                                   "without": float(fr.s["Dnoimg"].mean()), "gain": gain, "pass": gain > 0}}
        out = args.out or cfg["paths"]["runs"] / runs[-1] / "analysis_vision.json"
    else:
        sample = json.load(open(cfg["paths"]["samples"] / "stage1.json"))
        qset = set(sample["questions"])
        rel, cost = reliability_and_cost(cfg, runs)
        report = {"runs": runs, "ci": level, "bootstrap_reps": reps, "split_weighting": "equal"}

        fr = Frame(scores, ["A", "C"], qinfo, only=qset)
        report["1_main_C_vs_A"] = by_split_and_overall(fr, "A", "C", level, reps)
        groups = list(REF_GROUPS.values()) + ["multiple types"]
        report["2_by_reference_type"] = {g: compare(fr, "A", "C", level, reps, mask=fr.ref == g)
                                         for g in groups if (fr.ref == g).any()}

        if "O" in scores:
            fo = Frame(scores, ["O", "A"], qinfo, only=qset)
            report["3_text_loss_A_vs_O"] = by_split_and_overall(fo, "O", "A", level, reps)
        if "E" in scores:
            # Arm E: A's text plus MinerU's image analysis of each figure
            fe = Frame(scores, ["A", "C", "E"], qinfo, only=qset)
            report["7_figure_descriptions_E"] = {
                "E_vs_A": by_split_and_overall(fe, "A", "E", level, reps),
                "E_vs_C": by_split_and_overall(fe, "C", "E", level, reps),
                "E_vs_A_by_reference_type": {g: compare(fe, "A", "E", level, reps, mask=fe.ref == g)
                                             for g in groups if (fe.ref == g).any()},
            }
        if "G" in scores:
            # Arm G: DeepSeek-OCR-2 end to end (its own text, tables, captions and image readings)
            fg = Frame(scores, [a for a in ("A", "C", "E", "G") if a in scores], qinfo, only=qset)
            report["8_ocr2_pipeline_G"] = {
                "G_vs_A": by_split_and_overall(fg, "A", "G", level, reps),
                "G_vs_C": by_split_and_overall(fg, "C", "G", level, reps),
                **({"G_vs_E": by_split_and_overall(fg, "E", "G", level, reps)} if "E" in scores else {}),
                "G_vs_A_by_reference_type": {g: compare(fg, "A", "G", level, reps, mask=fg.ref == g)
                                             for g in groups if (fg.ref == g).any()},
            }
        if "N" in scores:
            fn = Frame(scores, ["N", "A", "C"], qinfo, only=sample["arm_n"])
            idx = fn.full_idx()
            report["4_no_paper"] = {"n": len(fn.qids),
                                    **{a: round(fn.weighted(fn.s[a], idx), 2) for a in ("N", "A", "C")}}
        report["5_reliability"] = rel
        report["6_cost"] = cost
        out = args.out or cfg["paths"]["runs"] / "analysis.json"
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
