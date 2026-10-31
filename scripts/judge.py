"""L3Score with Qwen3.8-27B (W4A16) on the 4090.

Start the server first (stop MinerU before):  bash scripts/serve_judge.sh
  python scripts/judge.py --run stage1          # scores runs/stage1/generations.jsonl
  python scripts/judge.py --check                # sanity check on 10 answer pairs

SPIQA's L3Score prompt, word for word. Thinking is off in the chat template so the first generated token
is the verdict; we read the Yes/No log-probabilities from the top-k of that first token and renormalize,
summing case and leading-space variants. Truncated, failed or empty answers score 0.
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.spiqa import load_all, load_config

# Verbatim from google/spiqa metrics/*_l3score.py
L3_PROMPT = ('You are given a question, ground-truth answer, and a candidate answer. Question: <question> \n'
             'Ground-truth answer: <GT> \nCandidate answer: <answer> \n'
             'Is the semantic meaning of the ground-truth and candidate answers similar? '
             'Answer in one word - Yes or No.')
YES, NO = {"yes", "yeah"}, {"no"}


def verdict_probs(top: list[dict]) -> tuple[float, float]:
    p_yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() in YES)
    p_no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() in NO)
    return p_yes, p_no


async def judge_one(client, sem, url, cfg, question, gt, answer):
    jc = cfg["judge"]
    prompt = L3_PROMPT.replace("<question>", question).replace("<GT>", gt).replace("<answer>", answer)
    async with sem:
        r = await client.post(url, timeout=600, json={
            "model": "judge", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": jc["top_logprobs"],
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]["content"][0]
    p_yes, p_no = verdict_probs(lp["top_logprobs"])
    score = p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.0
    return {"l3": score, "p_yes": p_yes, "p_no": p_no, "first_token": lp["token"],
            "verdict_mass": p_yes + p_no}


async def score_run(args, cfg):
    papers = load_all(cfg)
    qs = {q.qid: q for p in papers.values() for q in p.questions}
    run_dir = cfg["paths"]["runs"] / args.run
    gens = {}
    for line in open(run_dir / "generations.jsonl"):
        r = json.loads(line)
        if "error" not in r or r["id"] not in gens:
            gens[r["id"]] = r
    out_path = run_dir / "scores.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [g for g in gens.values() if g["id"] not in done]
    print(f"{len(todo)} to score ({len(done)} done)", flush=True)

    url = f"http://127.0.0.1:{cfg['judge']['port']}/v1/chat/completions"
    sem = asyncio.Semaphore(32)
    async with httpx.AsyncClient() as client:
        async def one(g):
            q = qs[g["qid"]]
            base = {"id": g["id"], "arm": g["arm"], "qid": g["qid"], "seed": g["seed"]}
            if "error" in g or g.get("truncated") or not g.get("answer"):
                why = "error" if "error" in g else ("truncated" if g.get("truncated") else "empty")
                return {**base, "l3": 0.0, "zeroed": why}
            return {**base, **await judge_one(client, sem, url, cfg, q.question, q.answer, g["answer"])}

        with open(out_path, "a") as f:
            for i, coro in enumerate(asyncio.as_completed([one(g) for g in todo])):
                f.write(json.dumps(await coro) + "\n")
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)


async def check(cfg):
    """Sanity check: 10 pairs (5 copies of the reference, 5 mismatched references)."""
    papers = load_all(cfg)
    qs = [q for p in papers.values() for q in p.questions if p.split == "C"][:10]
    url = f"http://127.0.0.1:{cfg['judge']['port']}/v1/chat/completions"
    sem = asyncio.Semaphore(10)
    rows = []
    async with httpx.AsyncClient() as client:
        for i, q in enumerate(qs):
            cand = q.answer if i < 5 else qs[(i + 3) % 10].answer
            r = await judge_one(client, sem, url, cfg, q.question, q.answer, cand)
            rows.append({"qid": q.qid, "expected": "Yes" if i < 5 else "No", **r})
            print(rows[-1], flush=True)
    ok_first = all(r["first_token"].strip().lower() in YES | NO for r in rows)
    ok_mass = all(r["verdict_mass"] > 0.9 for r in rows)
    ok_dir = all((r["l3"] >= 0.5) == (r["expected"] == "Yes") for r in rows)
    res = {"first_token_is_verdict": ok_first, "verdict_mass_gt_0.9": ok_mass, "direction_correct": ok_dir,
           "rows": rows}
    out = cfg["paths"]["runs"] / "phase0/check5_judge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print({k: v for k, v in res.items() if k != "rows"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    asyncio.run(check(cfg) if args.check else score_run(args, cfg))


if __name__ == "__main__":
    main()
