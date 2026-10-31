"""Which figures MinerU's advanced tier left undescribed (arm E), and what DeepSeek-OCR-2 (arm G) wrote for them.

  python scripts/dropped_descriptions.py     # -> runs/dropped_descriptions.json, runs/dropped_descriptions_samples.md

Each figure is one of the standard-tier figure blocks arm E builds on (panels regrouped under "Figure N"),
matched to the advanced run exactly as arm E does (intake.mineru.attach_descriptions). OCR-2 boxes are matched
to the same figure by page and box overlap (at least half of the smaller box). Score splits use the same
paper bootstrap as scripts/analyze.py.
"""

import copy
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.mineru import VISUAL, _overlap, attach_descriptions, load_blocks, paper_dir
from intake.prompts import label_of, spiqa_label
from intake.spiqa import ROOT, load_all, load_config
from scripts.analyze import Frame, compare, load_scores

REF = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.S)
RUNS = ["stage1", "stage1_flash", "stage1_e", "stage1_g"]


def raw_visuals(cfg, paper, out):
    """[(page, bbox, sub_type)] of every chart/image block in a MinerU run's middle JSON."""
    m = json.load(open(paper_dir(cfg, paper, out) / "raw/middle_json.json"))
    return [(p["page_idx"], b["bbox"], b.get("sub_type")) for p in m["pages"] for b in p["blocks"]
            if b.get("type") in VISUAL]


def ocr2_boxes(pages, figtext, stem, key):
    """{page: [(label, bbox 0-1, description or None, finish)]}, plus the set of dropped pages."""
    out, dropped = defaultdict(list), set()
    for (k, n), p in pages.items():
        if k != key:
            continue
        if p["finish"] != "stop":
            dropped.add(n)
            continue
        j = 0
        for label, coords in REF.findall(p["raw"]):
            try:
                boxes = eval(coords, {})
            except Exception:
                boxes = []
            for b, (x1, y1, x2, y2) in enumerate(boxes):
                desc = fin = None
                if label == "image":
                    r = figtext.get(f"data/ocr2_pipeline/crops/{stem}/p{n:02d}_img{j:02d}_{b}.png")
                    desc, fin = (r["text"], r["finish"]) if r else ("", None)
                out[n].append((label, [x1 / 999, y1 / 999, x2 / 999, y2 / 999], desc, fin))
            j += label == "image"
    return out, dropped


def eligible(bb):
    """MinerU's size rule for image analysis (mineru_vl_utils _is_eligible_for_image_analysis, 4.0.12)."""
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    return (w > 0.1 and h > 0.1) or w * h > 0.01


def describe_text(t):
    """Rough form of a figure reading: empty, table (with share of empty cells), mermaid, or prose/labels."""
    if not t.strip():
        return "empty", None
    if "<table" in t:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", t, re.S)
        empty = sum(not c.strip() for c in cells) / len(cells) if cells else 0.0
        return "table", round(empty, 2)
    if "mermaid" in t or re.search(r"^\s*(graph|flowchart) (TD|LR|TB)", t, re.M):
        return "mermaid", None
    return "text", None


def main():
    cfg = load_config()
    papers = load_all(cfg)
    stage = set(json.load(open(cfg["paths"]["samples"] / "stage1.json"))["questions"])
    pages = {}
    for l in open(ROOT / "data/ocr2_pipeline/pages.jsonl"):
        p = json.loads(l)
        pages[(p["paper"], p["page"])] = p
    figtext = {r["crop"]: r for r in map(json.loads, open(ROOT / "data/ocr2_pipeline/figures.jsonl"))}

    figs, fig_of = [], {}    # fig_of[(paper id, label)] -> index in figs
    for pid, paper in papers.items():
        qs = [q for q in paper.questions if q.qid in stage]
        if not qs:
            continue
        refs = Counter(spiqa_label(r) for q in qs for r in q.refs)
        spiqa_type = {spiqa_label(f.name): f.figure_type for f in paper.figures}
        std = load_blocks(cfg, paper)
        adv = load_blocks(cfg, paper, "mineru_adv_out")
        attach_descriptions(std, adv, label_of)
        std_raw, adv_raw = raw_visuals(cfg, paper, "mineru_out"), raw_visuals(cfg, paper, "mineru_adv_out")
        stem = paper_dir(cfg, paper).name
        boxes, dropped_pages = ocr2_boxes(pages, figtext, stem, pid)
        for b in std:
            if b.kind != "figure":
                continue
            lab = label_of(b.caption)
            panels = sum(pg == b.page and _overlap(bb, b.bbox) >= 0.5 for pg, bb, _ in std_raw)
            hits = [(bb, st) for pg, bb, st in adv_raw if pg == b.page and _overlap(bb, b.bbox) >= 0.5]
            adv_hits = [st for _, st in hits]
            if b.desc:
                why = "described"
            elif not adv_hits:
                why = "no box in advanced run"
            elif any(st is None and eligible(bb) for bb, st in hits):
                # big enough to analyse but no sub-type: only a collapsed image_block container ends up like this
                why = "folded multi-image container"
            elif all(st is None for st in adv_hits):
                why = "below size cutoff"
            else:
                why = "other"
            o2 = [x for x in boxes.get(b.page, []) if _overlap(x[1], b.bbox) >= 0.5]
            o2_img = [x for x in o2 if x[0] == "image"]
            o2_text = "\n\n".join(x[2] for x in o2_img if x[2])
            form, empty_share = describe_text(o2_text)
            m_form, m_empty = describe_text(b.desc)
            fig_of[(pid, lab)] = len(figs)
            figs.append({
                "paper": pid, "page": b.page, "label": lab, "caption": b.caption[:300],
                "spiqa_type": spiqa_type.get(lab), "referenced_by": refs.get(lab, 0) if lab else 0,
                "panels": panels, "adv_boxes": len(adv_hits), "mineru": why,
                "mineru_chars": len(b.desc), "mineru_form": m_form, "mineru_empty_cells": m_empty,
                "ocr2_page_dropped": b.page in dropped_pages,
                "ocr2_labels": sorted(Counter(x[0] for x in o2).items()),
                "ocr2_image_boxes": len(o2_img), "ocr2_chars": len(o2_text),
                "ocr2_cut_off": any(x[3] == "length" for x in o2_img),
                "ocr2_form": form, "ocr2_empty_cells": empty_share,
                "mineru_desc": b.desc, "ocr2_desc": o2_text,
            })

    # --- figure-level tallies
    def tally(rows):
        n = len(rows)
        out = {"figures": n}
        for k in ("mineru", "spiqa_type", "ocr2_form", "mineru_form"):
            out[k] = dict(Counter(str(r[k]) for r in rows).most_common())
        out["multi_panel"] = sum(r["panels"] > 1 for r in rows)
        out["referenced"] = sum(r["referenced_by"] > 0 for r in rows)
        out["ocr2_any_image_box"] = sum(r["ocr2_image_boxes"] > 0 for r in rows)
        out["ocr2_described"] = sum(r["ocr2_chars"] > 0 for r in rows)
        out["ocr2_page_dropped"] = sum(r["ocr2_page_dropped"] for r in rows)
        out["ocr2_cut_off"] = sum(r["ocr2_cut_off"] for r in rows)
        out["ocr2_labels_when_no_image_box"] = dict(Counter(
            "+".join(l for l, _ in r["ocr2_labels"]) or "nothing" for r in rows
            if r["ocr2_image_boxes"] == 0 and not r["ocr2_page_dropped"]).most_common())
        for src in ("mineru", "ocr2"):
            ch = [r[f"{src}_chars"] for r in rows if r[f"{src}_chars"]]
            out[f"{src}_chars_median"] = int(np.median(ch)) if ch else 0
            tabs = [r[f"{src}_empty_cells"] for r in rows if r[f"{src}_form"] == "table"]
            out[f"{src}_tables_mostly_empty"] = sum(e >= 0.5 for e in tabs)
        return out

    described = [f for f in figs if f["mineru"] == "described"]
    skipped = [f for f in figs if f["mineru"] != "described"]
    single = [f for f in figs if f["panels"] <= 1]
    multi = [f for f in figs if f["panels"] > 1]
    result = {
        "all": tally(figs), "described": tally(described), "skipped": tally(skipped),
        "skipped_referenced": tally([f for f in skipped if f["referenced_by"]]),
        "skip_rate": {
            "single_panel": [sum(f["mineru"] != "described" for f in single), len(single)],
            "multi_panel": [sum(f["mineru"] != "described" for f in multi), len(multi)],
            "by_spiqa_type": {t: [sum(f["mineru"] != "described" for f in figs if f["spiqa_type"] == t),
                                  sum(f["spiqa_type"] == t for f in figs)]
                              for t in sorted({str(f["spiqa_type"]) for f in figs})},
        },
    }
    # SPIQA types are keyed by str() above so None sorts; redo the counts with the real value
    for t in list(result["skip_rate"]["by_spiqa_type"]):
        tv = None if t == "None" else t
        result["skip_rate"]["by_spiqa_type"][t] = [
            sum(f["mineru"] != "described" for f in figs if f["spiqa_type"] == tv),
            sum(f["spiqa_type"] == tv for f in figs)]

    # --- question-level: did every non-table figure the question references reach E with a description?
    scores = load_scores(cfg, RUNS)
    qinfo, group = {}, {}
    for pid, paper in papers.items():
        for q in paper.questions:
            if q.qid not in stage:
                continue
            labs = {spiqa_label(r) for r in q.refs} - {None}
            figlabs = [l for l in labs if l.startswith("figure")]
            st = [figs[fig_of[(pid, l)]]["mineru"] == "described" for l in figlabs if (pid, l) in fig_of]
            if not figlabs:
                g = "tables only"
            elif len(st) < len(figlabs):
                g = "figure not found by MinerU"
            elif all(st):
                g = "all described"
            elif any(st):
                g = "some described"
            else:
                g = "none described"
            group[q.qid] = g
            qinfo[q.qid] = {"split": paper.split, "paper": paper.key, "ref_group": g}
    level, reps = cfg["analysis"]["ci"], cfg["analysis"]["bootstrap_reps"]
    fr = Frame(scores, ["A", "C", "E", "G"], qinfo)
    by_q = {}
    for g in ["tables only", "all described", "some described", "none described", "figure not found by MinerU"]:
        mask = fr.ref == g
        if not mask.any():
            continue
        by_q[g] = {f"{b}-A": compare(fr, "A", b, level, reps, mask=mask) for b in ("C", "E", "G")}
        by_q[g]["E-G"] = compare(fr, "G", "E", level, reps, mask=mask)
    result["questions_by_description_status"] = by_q

    out = cfg["paths"]["runs"]
    json.dump(result, open(out / "dropped_descriptions.json", "w"), indent=1)
    json.dump(figs, open(out / "dropped_descriptions_figures.json", "w"), indent=0)

    # --- samples to read by hand: skipped, referenced figures, with both readings side by side
    rng = random.Random(0)
    pick = rng.sample([f for f in skipped if f["referenced_by"]], 25)
    with open(out / "dropped_descriptions_samples.md", "w") as f:
        for r in pick:
            f.write(f"## {r['paper']} p{r['page']} {r['label']} ({r['spiqa_type']}, {r['panels']} panels, "
                    f"MinerU: {r['mineru']})\n\nCaption: {r['caption']}\n\n"
                    f"OCR-2 ({r['ocr2_form']}, {r['ocr2_chars']} chars, labels {r['ocr2_labels']}"
                    f"{', cut off' if r['ocr2_cut_off'] else ''}):\n\n```\n{r['ocr2_desc'][:1500]}\n```\n\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
