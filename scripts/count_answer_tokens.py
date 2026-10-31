"""Count each generation's answer tokens with the serving model's own tokenizer.

  python scripts/count_answer_tokens.py --run stage1 --model vision

The server reports completion tokens but not reasoning tokens, so reasoning tokens are derived later as
completion tokens minus answer tokens. Run this while the model that wrote the run is still being served.
Reads only generations.jsonl; writes runs/<run>/answer_tokens.jsonl (resumable).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.prompts import ARM_MODEL
from intake.spiqa import load_config


async def main_async(args):
    cfg = load_config()
    ep = cfg["endpoints"][args.model]
    url = ep["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize"
    run_dir = cfg["paths"]["runs"] / args.run
    out_path = run_dir / "answer_tokens.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    gens = [json.loads(l) for l in open(run_dir / "generations.jsonl")]
    todo = [g for g in gens if "error" not in g and g["id"] not in done and ARM_MODEL[g["arm"]] == args.model]
    print(f"{len(todo)} to count ({len(done)} done) with {ep['model']}", flush=True)

    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient() as client:
        async def one(g):
            async with sem:
                r = await client.post(url, json={"model": ep["model"], "prompt": g.get("content") or "",
                                                 "add_special_tokens": False}, timeout=60)
                r.raise_for_status()
                return {"id": g["id"], "arm": g["arm"], "qid": g["qid"], "answer_tokens": r.json()["count"],
                        "tokenizer": ep["model"]}
        rows = await asyncio.gather(*(one(g) for g in todo))
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", choices=["flash", "vision"], required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
