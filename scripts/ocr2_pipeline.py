"""Arm G's paper text: the whole DeepSeek-OCR-2 pipeline, raw PDF to text, MinerU not involved.

  1. pages    every page rendered at 144 DPI and read in document mode with layout boxes, as in the
              DeepSeek-OCR-2 repo's run_dpsk_ocr2_pdf.py (greedy, 20-gram no-repeat over 50 tokens,
              <td></td> exempt). Like that script, a page that never finishes (runs into the token limit
              while repeating) is dropped.
  2. figures  every box OCR-2 labels "image" is cropped from its page and read with DeepSeek-OCR's
              figure prompt.
  3. papers   the pages joined in order; layout tags removed; each image replaced by a placeholder plus
              OCR-2's reading of it, worded as in arms A and E. Captions, tables (HTML) and equations are
              OCR-2's own output.

Runs in the MinerU env (its vLLM supports DeepseekOCR2ForCausalLM) on the RTX 4090. Each stage resumes.

  envs/mineru/bin/python scripts/ocr2_pipeline.py [--papers 3]
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/ocr2_pipeline"
MODEL = ROOT / "models/DeepSeek-OCR-2"
REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
DPI = 144
PAGE_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
FIG_PROMPT = "<image>\nParse the figure."
WHITELIST = {128821, 128822}  # <td>, </td>
PAGE_NGRAM = dict(ngram_size=20, window_size=50, whitelist_token_ids=WHITELIST)
FIG_NGRAM = dict(ngram_size=20, window_size=90, whitelist_token_ids=WHITELIST)
REF = re.compile(r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.S)
EOS = "<｜end▁of▁sentence｜>"
FIG_TEXT = "[Figure: image not shown]{desc}"


def papers():
    manifest = json.load(open(ROOT / "data/pdfs/manifest.json"))
    keys = json.load(open(ROOT / "data/samples/stage1.json"))["papers"]
    return [(k, ROOT / "data/pdfs" / manifest[k]["file"]) for k in keys]


def render(pdf: Path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    return [doc[i].render(scale=DPI / 72).to_pil().convert("RGB") for i in range(len(doc))]


def done_keys(path: Path, key: str) -> set:
    return {json.loads(l)[key] for l in open(path)} if path.exists() else set()


def image_boxes(raw: str):
    """[(match text, [x1, y1, x2, y2] in 0-999 page coordinates)] for every 'image' layout box."""
    out = []
    for whole, label, coords in REF.findall(raw):
        if label == "image":
            try:
                boxes = eval(coords, {})
            except Exception:
                boxes = []
            out.append((whole, boxes))
    return out


def run_pages(llm, sp, todo):
    path = OUT / "pages.jsonl"
    for i in range(0, len(todo), 10):
        group = todo[i:i + 10]
        reqs, ids = [], []
        for key, pdf in group:
            for n, im in enumerate(render(pdf)):
                reqs.append({"prompt": PAGE_PROMPT, "multi_modal_data": {"image": im}})
                ids.append((key, n))
        res = llm.generate(reqs, sp, use_tqdm=False)
        with open(path, "a") as f:
            for (key, n), r in zip(ids, res):
                o = r.outputs[0]
                f.write(json.dumps({"paper": key, "page": n, "raw": o.text, "finish": o.finish_reason,
                                    "tokens": len(o.token_ids)}) + "\n")
        print(f"pages: {min(i + 10, len(todo))}/{len(todo)} papers", flush=True)


def run_figures(llm, sp, plist):
    """Crop every image box of every kept page, then read each crop."""
    pages = [json.loads(l) for l in open(OUT / "pages.jsonl")]
    done = done_keys(OUT / "figures.jsonl", "crop")
    by_paper = {}
    for p in pages:
        if p["finish"] == "stop":
            by_paper.setdefault(p["paper"], []).append(p)
    jobs = []
    for key, pdf in plist:
        if key not in by_paper:
            continue
        stem = pdf.stem
        rendered = None
        for p in by_paper[key]:
            for j, (_, boxes) in enumerate(image_boxes(p["raw"])):
                for b, (x1, y1, x2, y2) in enumerate(boxes):
                    crop = f"data/ocr2_pipeline/crops/{stem}/p{p['page']:02d}_img{j:02d}_{b}.png"
                    if crop in done:
                        continue
                    if not (ROOT / crop).exists():
                        rendered = rendered or render(pdf)
                        im = rendered[p["page"]]
                        w, h = im.size
                        box = (int(x1 / 999 * w), int(y1 / 999 * h), int(x2 / 999 * w), int(y2 / 999 * h))
                        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                            continue
                        (ROOT / crop).parent.mkdir(parents=True, exist_ok=True)
                        im.crop(box).save(ROOT / crop)
                    jobs.append(crop)
    print(f"figures: {len(jobs)} crops to read ({len(done)} already done)", flush=True)
    from PIL import Image
    for i in range(0, len(jobs), 64):
        batch = jobs[i:i + 64]
        reqs = [{"prompt": FIG_PROMPT, "multi_modal_data": {"image": Image.open(ROOT / c).convert("RGB")}}
                for c in batch]
        with open(OUT / "figures.jsonl", "a") as f:
            for c, r in zip(batch, llm.generate(reqs, sp, use_tqdm=False)):
                o = r.outputs[0]
                f.write(json.dumps({"crop": c, "text": o.text.replace(EOS, "").strip(),
                                    "finish": o.finish_reason, "tokens": len(o.token_ids)}) + "\n")
        print(f"figures: {min(i + 64, len(jobs))}/{len(jobs)}", flush=True)


def assemble(plist):
    pages = {}
    for l in open(OUT / "pages.jsonl"):
        p = json.loads(l)
        pages[(p["paper"], p["page"])] = p
    figs = OUT / "figures.jsonl"
    descs = {r["crop"]: r["text"] for r in map(json.loads, open(figs))} if figs.exists() else {}
    (OUT / "papers").mkdir(exist_ok=True)
    stats = {"papers": 0, "pages": 0, "pages_dropped": 0, "images": 0, "images_described": 0}
    for key, pdf in plist:
        parts, n = [], 0
        while (key, n) in pages:
            p = pages[(key, n)]
            n += 1
            stats["pages"] += 1
            if p["finish"] != "stop":
                stats["pages_dropped"] += 1
                continue
            content = p["raw"].replace(EOS, "")
            for j, (whole, boxes) in enumerate(image_boxes(p["raw"])):
                ds = [descs.get(f"data/ocr2_pipeline/crops/{pdf.stem}/p{p['page']:02d}_img{j:02d}_{b}.png", "")
                      for b in range(len(boxes))]
                ds = "\n\n".join(d for d in ds if d)
                stats["images"] += 1
                stats["images_described"] += bool(ds)
                desc = f"\nFigure content, read from the image by the OCR model:\n{ds}" if ds else ""
                content = content.replace(whole, FIG_TEXT.format(desc=desc) + "\n", 1)
            content = REF.sub("", content)
            content = content.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
            parts.append(re.sub(r"\n{3,}", "\n\n", content).strip())
        stats["papers"] += 1
        (OUT / "papers" / f"{pdf.stem}.md").write_text("\n\n".join(parts))
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print("assembled:", stats, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, help="first N papers only (trial)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plist = papers()[: args.papers]
    todo = [(k, p) for k, p in plist if k not in done_keys(OUT / "pages.jsonl", "paper")]
    print(f"pages: {len(todo)} papers to read", flush=True)

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    llm = LLM(model=str(MODEL), max_model_len=8192, gpu_memory_utilization=0.85, enable_prefix_caching=False,
              mm_processor_cache_gb=0, logits_processors=[NGramPerReqLogitsProcessor])
    # max_tokens as in the repo; generation still stops at the 8192-token context
    page_sp = SamplingParams(temperature=0.0, max_tokens=8192, extra_args=PAGE_NGRAM, skip_special_tokens=False)
    fig_sp = SamplingParams(temperature=0.0, max_tokens=8192, extra_args=FIG_NGRAM, skip_special_tokens=False)
    run_pages(llm, page_sp, todo)
    run_figures(llm, fig_sp, plist)
    assemble(plist)
    json.dump({"model": "deepseek-ai/DeepSeek-OCR-2", "revision": REVISION, "engine": "vllm " + vllm.__version__,
               "render": f"pypdfium2 at {DPI} DPI", "page_prompt": PAGE_PROMPT, "figure_prompt": FIG_PROMPT,
               "page_sampling": {"temperature": 0.0, "max_tokens": 8192, "max_model_len": 8192, "ngram_size": 20, "window_size": 50},
               "figure_sampling": {"temperature": 0.0, "max_tokens": 8192, "max_model_len": 8192, "ngram_size": 20, "window_size": 90},
               "whitelist_token_ids": sorted(WHITELIST), "unfinished_pages": "dropped, as in run_dpsk_ocr2_pdf.py"},
              open(OUT / "settings.json", "w"), indent=1)


if __name__ == "__main__":
    main()
