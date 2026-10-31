"""Download the arXiv PDFs for sampled papers at the listed version.

test-A keys carry a version (e.g. 1702.03584v3) and we fetch exactly that. test-B and test-C list only
bare arXiv IDs, so we take the latest version submitted before the source dataset existed: QASPER
(test-C, released 2021-05-07) and QASA (test-B, released 2023-07-01). The chosen version and the reason
are recorded in the manifest.
"""

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.spiqa import load_all, load_config, local_settings

CUTOFF = {"B": dt.date(2023, 7, 1), "C": dt.date(2021, 5, 7)}
# arXiv asks automated clients for a contact address; set CONTACT_EMAIL in .env
UA = {"User-Agent": f"spiqa-intake-test/0.1 (research; mailto:{local_settings().get('CONTACT_EMAIL', 'unset')})"}
DELAY_S = 3.0


def get(url: str) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {attempt + 1} {url}: {e}", flush=True)
            time.sleep(DELAY_S * 2 ** attempt)
    raise RuntimeError(f"failed: {url}")


def versions(arxiv_id: str) -> list[tuple[str, dt.date]]:
    html = get(f"https://arxiv.org/abs/{arxiv_id}").decode("utf-8", "replace")
    out = []
    for v, date in re.findall(r"\[(v\d+)\](?:</a>)?</strong>\s*([A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4})", html):
        out.append((v, dt.datetime.strptime(date, "%a, %d %b %Y").date()))
    if not out:
        raise RuntimeError(f"no submission history parsed for {arxiv_id}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    papers = load_all(cfg)
    sample = json.load(open(cfg["paths"]["samples"] / f"stage{args.stage}.json"))
    pdf_dir = cfg["paths"]["pdfs"]
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pdf_dir / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}

    for key in sample["papers"]:
        if key in manifest and (pdf_dir / manifest[key]["file"]).exists():
            continue
        p = papers[key]
        if p.arxiv_version:
            version, why = p.arxiv_version, "listed in SPIQA"
        else:
            hist = versions(p.arxiv_id)
            time.sleep(DELAY_S)
            before = [v for v, d in hist if d <= CUTOFF[p.split]]
            if before:
                version, why = before[-1], f"latest before {CUTOFF[p.split]}"
            else:
                version, why = hist[0][0], f"no version before {CUTOFF[p.split]}; took v1"
        name = f"{p.arxiv_id}{version}"
        data = get(f"https://arxiv.org/pdf/{name}")
        time.sleep(DELAY_S)
        if not data.startswith(b"%PDF"):
            raise RuntimeError(f"{name}: not a PDF ({data[:60]!r})")
        fname = f"{p.split}_{name}.pdf"
        (pdf_dir / fname).write_bytes(data)
        manifest[key] = {"arxiv": name, "version_rule": why, "file": fname,
                         "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"{key} -> {name} ({why}, {len(data) // 1024} KB)", flush=True)

    print(f"{sum(k in manifest for k in sample['papers'])}/{len(sample['papers'])} PDFs present")


if __name__ == "__main__":
    main()
