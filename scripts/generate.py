"""Send frozen arm inputs to the DeepSeek endpoints on the Sparks; interleaved, resumable, fully logged.

  python scripts/generate.py --run stage1 --set questions --arms A,B,C,D,O
  python scripts/generate.py --run stage1 --set arm_n --arms N
  python scripts/generate.py --run pilot --set flip_pilot --arms A,C
  python scripts/generate.py --run stage1_flash --arms A,O,N --arm-set N=arm_n   # N on its own subset

Jobs across all requested arms are shuffled together with a fixed seed so load and drift cannot line
up with an arm. With --model flash|vision only that model's arms run (one-model-at-a-time fallback);
arms are still interleaved within the block. Each result line logs tokens (text/image/output/reasoning),
latency, finish reason and the serving fingerprint.
"""

import argparse
import asyncio
import base64
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intake.prompts import ARM_MODEL
from intake.spiqa import ROOT, load_config


def encode(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = []
        for c in m["content"]:
            if c["type"] == "image_path":
                b64 = base64.b64encode((ROOT / c["path"]).read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            else:
                content.append(c)
        out.append({"role": m["role"], "content": content})
    return out


def text_only(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": "".join(c["text"] for c in m["content"] if c["type"] == "text")}
            for m in messages]


ANSWER_RE = re.compile(r"""['"]Answer['"]\s*:\s*(['"])(.*)\1\s*}""", re.S)


def parse_answer(content: str | None) -> tuple[str, bool]:
    """Return (answer, parsed_ok). Falls back to the raw content when the SPIQA format is missing."""
    if not content:
        return "", False
    m = ANSWER_RE.search(content)
    if m:
        return m.group(2).strip(), True
    return content.strip(), False


async def count_text_tokens(client, ep, messages) -> int | None:
    """Text tokens of the same input without images, so image tokens = prompt - text."""
    try:
        r = await client.post(ep["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize",
                              json={"model": ep["model"], "messages": text_only(messages)}, timeout=120)
        return r.json().get("count")
    except Exception:
        return None


async def run_job(client, sem, ep, gen, job, out_f, lock):
    inp = json.load(open(job["input_path"]))
    body = {
        "model": ep["model"],
        "messages": encode(inp["messages"]),
        "temperature": gen["temperature"],
        "top_p": gen["top_p"],
        "max_tokens": gen["max_output_tokens"],
        "seed": job["seed"],
        **ep.get("extra_body", {}),
    }
    async with sem:
        t0 = time.time()
        err, resp = None, None
        for attempt in range(3):
            try:
                r = await client.post(ep["base_url"].rstrip("/") + "/chat/completions", json=body,
                                      timeout=gen["request_timeout_s"])
                r.raise_for_status()
                resp = r.json()
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(10 * (attempt + 1))
        latency = time.time() - t0
        has_images = any(c["type"] == "image_path" for m in inp["messages"] for c in m["content"])
        n_text = await count_text_tokens(client, ep, inp["messages"]) if has_images and resp else None

    rec = {**{k: job[k] for k in ("id", "arm", "qid", "seed")}, "input_sha256": inp["input_sha256"],
           "endpoint": ep["model"], "latency_s": round(latency, 2), "ts": time.time()}
    if resp is None:
        rec.update(error=err)
    else:
        ch = resp["choices"][0]
        msg = ch["message"]
        usage = resp.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        answer, ok = parse_answer(msg.get("content"))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        prompt_tokens = usage.get("prompt_tokens")
        rec.update(
            finish_reason=ch.get("finish_reason"),
            truncated=ch.get("finish_reason") == "length",
            content=msg.get("content"), answer=answer, parsed=ok,
            reasoning_chars=len(reasoning),
            prompt_tokens=prompt_tokens,
            text_tokens=n_text if has_images else prompt_tokens,
            image_tokens=(prompt_tokens - n_text) if (has_images and n_text and prompt_tokens) else 0,
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            system_fingerprint=resp.get("system_fingerprint"),
            reasoning=reasoning,
        )
    async with lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
    status = rec.get("error") or f"{rec['finish_reason']} {rec['completion_tokens']}tok"
    print(f"{job['id']}  {latency:6.1f}s  {status}", flush=True)


async def main_async(args):
    cfg = load_config()
    gen = cfg["generation"]
    sample = json.load(open(cfg["paths"]["samples"] / f"stage{args.stage}.json"))
    arm_sets = dict(kv.split("=") for kv in args.arm_set.split(",")) if args.arm_set else {}
    arms = args.arms.split(",")
    if args.model:
        arms = [a for a in arms if ARM_MODEL[a] == args.model]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else gen["seeds"]

    run_dir = cfg["paths"]["runs"] / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "generations.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["id"])

    # Shuffle (arm, question) units so arms interleave, but keep a question's seeds adjacent: they share
    # the whole prompt, so the server's prefix cache prefills it once instead of three times.
    units = []
    for arm in arms:
        for qid in sample[arm_sets.get(arm, args.set)]:
            ip = cfg["paths"]["inputs"] / arm / (qid.replace("/", "__") + ".json")
            if not ip.exists():
                sys.exit(f"missing frozen input {ip}; run scripts/build_inputs.py first")
            unit = [{"id": f"{arm}|{qid}|{s}", "arm": arm, "qid": qid, "seed": s, "input_path": str(ip)}
                    for s in seeds if f"{arm}|{qid}|{s}" not in done]
            if unit:
                units.append(unit)
    random.Random(f"{args.run}-order").shuffle(units)
    jobs = [j for u in units for j in u]
    print(f"{len(jobs)} jobs to run ({len(done)} already done) -> {out_path}", flush=True)

    eps = {}
    for m in {ARM_MODEL[a] for a in arms}:
        ep = cfg["endpoints"][m]
        if not ep.get("base_url"):
            sys.exit("DEEPSEEK_BASE_URL not set in .env")
        eps[m] = ep
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump({"args": vars(args), "generation": gen, "endpoints": eps, "started": time.time()}, f, indent=1)

    sem, lock = asyncio.Semaphore(gen["concurrency"]), asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with open(out_path, "a") as out_f:
            await asyncio.gather(*(run_job(client, sem, eps[ARM_MODEL[j["arm"]]], gen, j, out_f, lock)
                                   for j in jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--set", default="questions",
                    help="key in the sample file: questions, arm_n, flip_pilot, dry_run, vision_check")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--arm-set", help="per-arm sample key overriding --set, e.g. N=arm_n")
    ap.add_argument("--model", choices=["flash", "vision"])
    ap.add_argument("--seeds", help="override, e.g. 0")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
