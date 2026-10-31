"""Load the three SPIQA test splits into one question/paper schema."""

import json
import re
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = ROOT / v
    base_url = local_settings().get("DEEPSEEK_BASE_URL", "")
    for ep in cfg["endpoints"].values():
        ep["base_url"] = base_url
    return cfg


def local_settings() -> dict:
    """KEY=value lines from .env, which holds machine-specific settings and stays out of the repo."""
    env = ROOT / ".env"
    if not env.exists():
        return {}
    lines = (l.strip() for l in env.read_text().splitlines())
    return dict(l.split("=", 1) for l in lines if l and not l.startswith("#"))


@dataclass
class Figure:
    name: str            # SPIQA file name as referenced by questions
    path: str            # full-resolution image path, relative to ROOT
    caption: str
    is_table: bool
    figure_type: str     # normalized: table, plot, schematic, photo, other


@dataclass
class Question:
    qid: str             # globally unique: <split>/<paper_key>/<index>
    split: str
    paper_key: str
    question: str
    answer: str
    refs: list[str]
    is_table_q: bool     # any reference is a table
    ref_types: list[str]


@dataclass
class Paper:
    split: str
    key: str             # SPIQA dict key
    arxiv_id: str        # without version
    arxiv_version: str | None  # e.g. "v3"; only test-A lists one
    title: str
    figures: list[Figure]
    clean_text: str      # SPIQA's extracted paragraphs (arm O)
    questions: list[Question] = field(default_factory=list)

    @property
    def lean(self) -> str:
        """Majority reference type across the paper's questions."""
        n_tab = sum(t == "table" for q in self.questions for t in q.ref_types)
        n_all = sum(len(q.ref_types) for q in self.questions)
        return "table" if n_tab * 2 >= n_all else "figure"

    def to_dict(self):
        d = asdict(self)
        d["lean"] = self.lean
        return d


def norm_type(raw: str, content_type: str | None = None) -> str:
    t = re.sub(r"[^a-z() ]", " ", (raw or "").lower()).strip()
    if content_type == "table" or t.startswith("table"):
        return "table"
    if "plot" in t:
        return "plot"
    if "schematic" in t:
        return "schematic"
    if "photo" in t:
        return "photo"
    return "other"


def _load_a(spiqa: Path) -> list[Paper]:
    data = json.load(open(spiqa / "test-A/SPIQA_testA.json"))
    papers = []
    for key, p in data.items():
        m = re.fullmatch(r"(.+?)(v\d+)", key)
        figs = []
        for name, f in p["all_figures"].items():
            ft = norm_type(f["figure_type"], f["content_type"])
            figs.append(Figure(name, f"data/spiqa/test-A/SPIQA_testA_Images/{key}/{name}",
                               f["caption"], ft == "table", ft))
        text = (spiqa / "SPIQA_train_val_test-A_extracted_paragraphs" / f"{key}.txt").read_text()
        paper = Paper("A", key, m.group(1), m.group(2), "", figs, text)
        by_name = {f.name: f for f in figs}
        for i, qa in enumerate(p["qa"]):
            rt = [by_name[qa["reference"]].figure_type]
            paper.questions.append(Question(f"A/{key}/{i}", "A", key, qa["question"], qa["answer"],
                                            [qa["reference"]], rt[0] == "table", rt))
        papers.append(paper)
    return papers


def _load_b(spiqa: Path) -> list[Paper]:
    data = json.load(open(spiqa / "test-B/SPIQA_testB.json"))
    papers = []
    for key, p in data.items():
        figs = []
        for name, caption in p["all_figures_tables"].items():
            ft = norm_type(p["figure_types"][name])
            figs.append(Figure(name, f"data/spiqa/test-B/SPIQA_testB_Images/{name}", caption,
                               ft == "table", ft))
        paper = Paper("B", key, key, None, p["title"], figs, "\n\n".join(p["passages"]))
        by_name = {f.name: f for f in figs}
        for i, q in enumerate(p["question"]):
            refs = sorted(set(p["referred_figures_tables"][i]))
            rt = [by_name[r].figure_type for r in refs]
            paper.questions.append(Question(f"B/{key}/{p['question_key'][i]}", "B", key, q,
                                            p["composition"][i], refs, "table" in rt, rt))
        papers.append(paper)
    return papers


def _load_c(spiqa: Path) -> list[Paper]:
    data = json.load(open(spiqa / "test-C/SPIQA_testC.json"))
    papers = []
    for key, p in data.items():
        figs = []
        for f in p["figures_and_tables"]:
            ft = norm_type(p["figure_types"][f"{key}/{f['file']}"])
            figs.append(Figure(f["file"], f"data/spiqa/test-C/SPIQA_testC_Images/{key}/{f['file']}",
                               f["caption"], ft == "table", ft))
        # Same construction as SPIQA's own test-C full-text eval
        text = "\n\n".join("\n".join(s["paragraphs"]) for s in p["full_text"])
        paper = Paper("C", key, key, None, p["paper_title"], figs, text)
        by_name = {f.name: f for f in figs}
        for i, q in enumerate(p["question"]):
            a = p["answer"][i]
            ans = a["free_form_answer"] or {True: "Yes", False: "No"}[a["yes_no"]]
            refs = sorted(set(p["referred_figures_tables"][i]))
            rt = [by_name[r].figure_type for r in refs]
            paper.questions.append(Question(f"C/{key}/{p['question_key'][i]}", "C", key, q, ans,
                                            refs, "table" in rt, rt))
        papers.append(paper)
    return papers


def load_all(cfg: dict | None = None) -> dict[str, Paper]:
    """All test papers keyed by '<split>/<key>'."""
    cfg = cfg or load_config()
    spiqa = cfg["paths"]["spiqa"]
    out = {}
    for p in _load_a(spiqa) + _load_b(spiqa) + _load_c(spiqa):
        out[f"{p.split}/{p.key}"] = p
    return out
