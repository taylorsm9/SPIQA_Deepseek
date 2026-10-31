"""Download SPIQA's test data into data/spiqa and the local models into models/, at the revisions we used.

  python scripts/fetch_data.py              # SPIQA, all three models, and MinerU's small models
  python scripts/fetch_data.py --only spiqa

The arXiv PDFs come separately, from scripts/fetch_pdfs.py. The DeepSeek V4 Flash models are served
remotely and are not downloaded here.
"""

import argparse
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.spiqa import load_config

ROOT = Path(__file__).resolve().parent.parent

SPIQA = ("google/spiqa", "1774b71511f029b82089a069d75328f25fbf0705")
SPIQA_FILES = ["SPIQA_train_val_test-A_extracted_paragraphs.zip",
               "test-A/SPIQA_testA.json", "test-A/SPIQA_testA_Images.zip",
               "test-B/SPIQA_testB.json", "test-B/SPIQA_testB_Images.zip",
               "test-C/SPIQA_testC.json", "test-C/SPIQA_testC_Images.zip"]

MODELS = {
    "mineru": ("opendatalab/MinerU2.5-Pro-2605-1.2B", "08aaea840498d49ce16247b6263196cf02814885"),
    "ocr2": ("deepseek-ai/DeepSeek-OCR-2", "aaa02f3811945a91062062994c5c4a3f4c0af2b0"),
    "judge": ("Vishva007/Qwen3.8-27B-W4A16-AutoRound", "d31b835986575b27eda14d05f21e1f7f68610c6d"),
}
# MinerU's layout, formula, OCR and table models, which it looks for under $MINERU_HOME/models
MINERU_SMALL = ("opendatalab/MinerU-4_models_torch", "2b3afb86f4d23fa623f6b2f4b2279ed4ef93a89d")


def fetch_spiqa(dest: Path):
    repo, rev = SPIQA
    snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=SPIQA_FILES, local_dir=dest)
    for name in SPIQA_FILES:
        if name.endswith(".zip"):
            z = dest / name
            if not (z.parent / z.stem).exists():   # each zip holds one folder named after it
                zipfile.ZipFile(z).extractall(z.parent)
                print(f"unpacked {z.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["spiqa", *MODELS])
    args = ap.parse_args()
    cfg = load_config()
    if args.only in (None, "spiqa"):
        fetch_spiqa(cfg["paths"]["spiqa"])
    for key, (repo, rev) in MODELS.items():
        if args.only in (None, key):
            snapshot_download(repo, revision=rev, local_dir=ROOT / "models" / repo.split("/")[1])
            print(f"downloaded {repo} @ {rev[:8]}")
    if args.only in (None, "mineru"):
        fetch_mineru_home()


def fetch_mineru_home():
    """Lay out mineru_home/models the way MinerU expects: its small models, plus a link to the VLM."""
    models = ROOT / "mineru_home/models"
    repo, rev = MINERU_SMALL
    snapshot_download(repo, revision=rev, local_dir=models / repo.split("/")[1])
    vlm = models / MODELS["mineru"][0].split("/")[1]
    if not vlm.exists():
        vlm.symlink_to(Path("../../models") / vlm.name)
    print(f"downloaded {repo} @ {rev[:8]} into {models.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
