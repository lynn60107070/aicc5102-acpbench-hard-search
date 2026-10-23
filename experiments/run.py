"""Run one method on ACPBench-Hard Next Action and score it (owner: Aisha).

    python -m experiments.run --method direct --limit 10
    python -m experiments.run --method search --param k=3 --param d=2 --max-calls 12 --seed 1
    python -m experiments.run --method search --limit 10 --workers 4

Writes results/runs/<name>/rows.jsonl (one row per instance) and summary.json.
"""

import argparse
import ast
import json
import math
import re
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import validator
from common.data import load_docs, to_instance
from common.interfaces import Budget, BudgetExceeded
from common.llm import REPO_ROOT, LLMClient
from methods import METHODS


def parse_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value  # plain strings like scoring=random
    return params


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    half = 1.96 * math.sqrt(acc * (1 - acc) / n)
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r["correct"])
    return {
        "n": n,
        "accuracy": acc,
        "ci95": [max(0.0, acc - half), min(1.0, acc + half)],
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "accuracy_by_domain": {
            d: sum(v) / len(v) for d, v in sorted(by_domain.items())
        },
        "total_calls": sum(r["calls"] for r in rows),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
        "mean_calls": sum(r["calls"] for r in rows) / n,
        "mean_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / n,
        "providers": dict(Counter(p for r in rows for p in r.get("providers", []))),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--param", action="append", default=[], help="method parameter, e.g. k=3")
    ap.add_argument("--model", help="overrides ACP_MODEL")
    ap.add_argument("--base-url", help="overrides ACP_BASE_URL")
    ap.add_argument("--seed", type=int, default=1234, help="1234 = the seed lm-eval uses")
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "validation"],
        help="tune on validation (40), report on test (130)",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of questions to process concurrently",
    )
    ap.add_argument("--ids", nargs="*", help="run only these instance ids")
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--name", help="output folder name (default derived from the settings)")
    args = ap.parse_args(argv)

    if args.workers < 1:
        ap.error("--workers must be at least 1")

    params = parse_params(args.param)
    client = LLMClient.from_env(model=args.model, base_url=args.base_url, seed=args.seed)
    solve = METHODS[args.method]

    name = args.name or "_".join(
        [args.method, re.sub(r"[^\w.-]", "-", client.model), f"s{args.seed}"]
        + [f"{k}{v}" for k, v in sorted(params.items())]
        + ([f"lim{args.limit}"] if args.limit else [])
        + (["val"] if args.split == "validation" else [])
    )
    out = REPO_ROOT / "results" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    docs = load_docs(limit=args.limit, ids=args.ids, split=args.split)
    rows = []

    def run_one(item):
        i, doc = item
        instance = to_instance(doc)

        # Each question must have its own session and budget.
        llm = client.session(
            Budget(max_calls=args.max_calls, max_tokens=args.max_tokens)
        )
        start = time.time()
        response, crash, exhausted = "", None, False

        try:
            response = solve(instance, llm, **params)
        except BudgetExceeded:
            exhausted = True  # methods should catch this themselves and return their best guess
        except Exception:
            crash = traceback.format_exc()
            print(f"  ! {instance.id} crashed:\n{crash}")

        verdict = validator.score(doc, response)
        row = {
            "id": instance.id,
            "domain": instance.domain,
            "method": args.method,
            "params": params,
            "seed": args.seed,
            "model": client.model,
            "response": response,
            "action": verdict.action,
            "correct": verdict.correct,
            "error": "crash" if crash else verdict.error,
            "budget_exhausted": exhausted,
            "calls": llm.budget.calls,
            "prompt_tokens": llm.budget.prompt_tokens,
            "completion_tokens": llm.budget.completion_tokens,
            "calls_by_tag": llm.budget.calls_by_tag,
            "tokens_by_tag": llm.budget.tokens_by_tag,
            "providers": sorted({str(c.get("provider")) for c in llm.log}),
            # Every reply (no prompts), for analysis, e.g. whether a correct candidate was proposed.
            "replies": [{"tag": c["tag"], "text": c["text"]} for c in llm.log],
            "seconds": round(time.time() - start, 2),
            "crash": crash,
        }
        return i, instance, row

    with open(out / "rows.jsonl", "w", encoding="utf8") as f:
        # executor.map preserves input order in the written output.
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for i, instance, row in executor.map(run_one, enumerate(docs, 1)):
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()

                acc = sum(r["correct"] for r in rows) / len(rows)
                status = "OK " if row["correct"] else row["error"] or "X"
                print(
                    f"[{i}/{len(docs)}] {instance.domain:<18} {status:<12}"
                    f" {str(row['action'])[:40]:<40} running acc {acc:.3f}",
                    flush=True,
                )

    summary = {
        "settings": {**vars(args), "params": params, "model": client.model},
        **summarize(rows),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
