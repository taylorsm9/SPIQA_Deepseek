"""Build the frozen per-setup inputs. The code calls each setup by a letter:

Every arm uses SPIQA's full-paper direct-QA prompt, unchanged. Messages are stored with image file
paths; images are base64-encoded only at send time, so the frozen inputs stay small and hashable.

  A  V4-Flash        MinerU text, tables as HTML, figures as caption + placeholder
  C  Vision-Exp      A's text, plus MinerU crops inline where MinerU found them, each with its caption
  D  Vision-Exp      captions only + SPIQA's full-resolution images
  E  V4-Flash        A's text exactly, plus MinerU's image analysis of each figure (advanced tier) under its caption
  G  V4-Flash        DeepSeek-OCR-2 end to end: its own page text, tables and captions, plus its reading of each image
  O  V4-Flash        SPIQA's clean paragraphs + the same MinerU tables A sees + figure captions
  N  V4-Flash        question only
  Dnoimg             D without images. D and Dnoimg are only for the check that images reach the model

The server accepts at most 32 images per prompt (tiles count). When a paper needs more, the question's
referenced figures get their images first, then the rest in paper order; figures past the cap fall back
to caption only, and the input records how many were dropped.
"""

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from .mineru import Block, attach_descriptions, load_blocks, paper_dir
from .spiqa import ROOT, Paper, Question

# Verbatim from google/spiqa evals/*/gpt4v_qa_test-*_evaluation_image+caption+full_text.py
SPIQA_PROMPT = ("You are given a question, paragraphs from a scientific paper, a few input images, and a caption "
                "corresponding to each input image. Please answer the question based on the paper, input images "
                "and corresponding captions. Question: <question>. Output in the following format: "
                "{'Answer': 'Direct Answer to the Question'}. \n")

ARM_MODEL = {"A": "flash", "C": "vision", "D": "vision", "E": "flash", "G": "flash", "O": "flash",
             "N": "flash", "Dnoimg": "vision"}
ARM_NEEDS_OCR = {"A", "C", "E", "O"}
LABEL = re.compile(r"(fig(?:ure)?|table)\.?\s*(\d+)", re.I)


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def image(path: str) -> dict:
    return {"type": "image_path", "path": path}


def label_of(caption: str) -> str | None:
    """'Figure 3: ...' -> 'figure3'; 'Table 2 (b)' -> 'table2'."""
    m = LABEL.match(caption.strip()) if caption else None
    return f"{'table' if m.group(1).lower() == 'table' else 'figure'}{m.group(2)}" if m else None


def spiqa_label(name: str) -> str | None:
    """SPIQA file names look like '1611.04684v1-Table3-1.png' or '4-Figure2-1.png'."""
    m = re.search(r"(Figure|Table)(\d+)-\d+\.png$", name)
    return f"{m.group(1).lower()}{m.group(2)}" if m else None


def tile(path: Path, out_dir: Path, tall_ratio: float, overlap: float) -> list[str]:
    """Split a crop taller than tall_ratio x width into overlapping tiles of about width x tall_ratio."""
    im = Image.open(path)
    w, h = im.size
    if h <= tall_ratio * w:
        return [str(path.relative_to(ROOT))]
    tile_h = int(tall_ratio * w)
    step = int(tile_h * (1 - overlap))
    out, top, i = [], 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        bottom = min(top + tile_h, h)
        p = out_dir / f"{path.stem}_tile{i}.png"
        if not p.exists():
            im.crop((0, top, w, bottom)).save(p)
        out.append(str(p.relative_to(ROOT)))
        if bottom >= h:
            return out
        top += step
        i += 1


def allot_images(items: list[tuple[list[str], bool]], cap: int) -> list[bool]:
    """items: (tile paths, is_referenced) in paper order. Returns which items keep their images."""
    keep = [False] * len(items)
    used = 0
    for prio in (True, False):
        for i, (tiles, ref) in enumerate(items):
            if ref == prio and used + len(tiles) <= cap:
                keep[i] = True
                used += len(tiles)
    return keep


def ocr_parts(blocks: list[Block], with_crops: bool, refs: set[str], cfg: dict, tile_dir: Path):
    """MinerU blocks in reading order. Tables as caption + HTML; figures as caption + placeholder.
    With crops, each figure/table crop goes inline right after its caption, labelled like SPIQA's images."""
    img_cfg = cfg["images"]
    visual = [b for b in blocks if b.kind in ("figure", "table") and b.crop]
    tiles = {id(b): tile(ROOT / b.crop, tile_dir, img_cfg["tall_ratio"], img_cfg["tile_overlap"])
             for b in visual} if with_crops else {}
    keep = dict(zip([id(b) for b in visual],
                    allot_images([(tiles[id(b)], label_of(b.caption) in refs) for b in visual],
                                 cfg["endpoints"]["vision"]["max_images"]))) if with_crops else {}

    parts, buf, n_img = [], [], 0

    def flush():
        if buf:
            parts.append(text("\n\n".join(buf)))
            buf.clear()

    for b in blocks:
        if b.kind == "text":
            buf.append(b.text)
            continue
        cap = b.caption or "(no caption)"
        if with_crops and keep.get(id(b)):
            flush()
            for tp in tiles[id(b)]:
                parts += [text(f"Image {n_img}: "), image(tp)]
            parts.append(text(f"Caption {n_img}: {cap} \n\n"))
            n_img += 1
            if b.kind == "table" and b.html:
                buf.append(b.html)
        elif b.kind == "table":
            buf.append(f"[Table] {cap}\n{b.html or '(table content not recognized)'}")
        else:
            desc = f"\nFigure content, read from the image by the OCR model:\n{b.desc}" if b.desc else ""
            buf.append(f"[Figure: image not shown] {cap}{desc}")
    flush()
    dropped = sum(not v for v in keep.values())
    ref_found = sum(label_of(b.caption) in refs for b in visual)
    return parts, dropped, ref_found


def build(arm: str, q: Question, paper: Paper, cfg: dict) -> dict:
    content = [text(SPIQA_PROMPT.replace("<question>", q.question))]
    img_cfg = cfg["images"]
    refs = {spiqa_label(r) for r in q.refs} - {None}
    meta = {}
    if arm in ARM_NEEDS_OCR:
        blocks = load_blocks(cfg, paper)
    if arm == "E":
        meta["figures_described"] = attach_descriptions(blocks, load_blocks(cfg, paper, "mineru_adv_out"), label_of)
    if arm in ("A", "C", "E"):
        tile_dir = cfg["paths"]["inputs"] / "tiles" / paper.split / paper.key.replace("/", "_")
        body, dropped, ref_found = ocr_parts(blocks, arm == "C", refs, cfg, tile_dir)
        content.append(text("Paragraphs from the paper: "))
        content += body
        content.append(text(": "))
        meta = {**meta, "images_dropped": dropped, "refs_matched_in_ocr": ref_found, "refs": len(refs)}
    elif arm == "G":
        # scripts/ocr2_pipeline.py; the same wrapper text as A around OCR-2's whole paper
        md = cfg["paths"]["ocr2_pipeline"] / "papers" / f"{paper_dir(cfg, paper).name}.md"
        content.append(text(f"Paragraphs from the paper: {md.read_text()}: "))
    elif arm == "O":
        tables = [f"[Table] {b.caption or '(no caption)'}\n{b.html or '(table content not recognized)'}"
                  for b in blocks if b.kind == "table"]
        caps = [f"[Figure: image not shown] {f.caption}" for f in paper.figures if not f.is_table]
        content.append(text("Paragraphs from the paper: {}: ".format(
            "\n\n".join([paper.clean_text] + tables + caps))))
    elif arm in ("D", "Dnoimg"):
        tdir = cfg["paths"]["inputs"] / "tiles_spiqa" / paper.split
        tiles = [tile(ROOT / f.path, tdir, img_cfg["tall_ratio"], img_cfg["tile_overlap"])
                 for f in paper.figures] if arm == "D" else []
        keep = allot_images([(t, f.name in q.refs) for t, f in zip(tiles, paper.figures)],
                            cfg["endpoints"]["vision"]["max_images"]) if arm == "D" else []
        for i, f in enumerate(paper.figures):
            if arm == "D" and keep[i]:
                for tp in tiles[i]:
                    content += [text(f"Image {i}: "), image(tp)]
            content.append(text(f"Caption {i}: {f.caption} \n\n"))
        meta = {"images_dropped": sum(not k for k in keep)} if arm == "D" else {}
    elif arm != "N":
        raise ValueError(arm)
    msgs = [{"role": "user", "content": merge_text(content)}]
    blob = json.dumps(msgs, sort_keys=True).encode()
    for c in msgs[0]["content"]:
        if c["type"] == "image_path":
            blob += hashlib.sha256((ROOT / c["path"]).read_bytes()).digest()
    return {"arm": arm, "qid": q.qid, "model": ARM_MODEL[arm], "messages": msgs,
            "n_images": sum(c["type"] == "image_path" for c in msgs[0]["content"]), **meta,
            "input_sha256": hashlib.sha256(blob).hexdigest()}


def merge_text(content: list[dict]) -> list[dict]:
    out = []
    for c in content:
        if c["type"] == "text" and out and out[-1]["type"] == "text":
            out[-1] = text(out[-1]["text"] + c["text"])
        else:
            out.append(c)
    return out
