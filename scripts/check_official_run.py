"""Check an official lm-eval reproduction run and compare it with our reference run.

    python scripts/check_official_run.py results/official_reproduction

Confirms the folder came from the official lm-eval harness (not experiments.run), used the paper's
settings on all 130 questions, and reports per-question agreement with
results/final/repro_llama31_8b_direct_bf16_coreweave.
"""

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = REPO_ROOT / "results/final/repro_llama31_8b_direct_bf16_coreweave/rows.jsonl"


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'OK' if ok else '!!'}] {label}: {detail}")
    return ok


def main(folder: str) -> int:
    results = sorted(glob.glob(str(Path(folder) / "**" / "results_*.json"), recursive=True))
    samples = sorted(glob.glob(str(Path(folder) / "**" / "samples_acp_nexta_gen_*.jsonl"), recursive=True))
    if not results or not samples:
        print("No lm-eval results_*.json / samples_acp_nexta_gen_*.jsonl found: this is not an official "
              "harness run (experiments.run writes summary.json + rows.jsonl instead).")
        return 1
    r = json.loads(Path(results[-1]).read_text(encoding="utf8"))
    rows = [json.loads(line) for line in Path(samples[-1]).read_text(encoding="utf8").splitlines()]
    # For API models lm-eval leaves chat_template empty; the requests themselves show it was applied.
    first_request = rows[0]["arguments"]["gen_args_0"]["arg_0"][0]
    chat = first_request.lstrip().startswith('[{"role"')
    cfg = r["configs"]["acp_nexta_gen"]
    gen = cfg["generation_kwargs"]
    score = r["results"]["acp_nexta_gen"]["score,acp_grammar_parse"]
    n = r["n-samples"]["acp_nexta_gen"]["effective"]

    print(f"Checking {results[-1]}")
    ok = all([
        check("tool", "lm_eval_version" in r, f"official lm-eval harness {r.get('lm_eval_version')}"),
        check("model", r["config"]["model_args"].get("model") == "meta-llama/llama-3.1-8b-instruct",
              str(r["config"]["model_args"])),
        check("questions", n == 130, f"{n} of 130"),
        check("dataset", cfg["dataset_path"] == "ibm-research/acp_bench" and cfg["test_split"] == "test",
              f"{cfg['dataset_path']} / {cfg['test_split']}"),
        check("2-shot", cfg["num_fewshot"] == 2, str(cfg["num_fewshot"])),
        check("greedy", gen.get("temperature") == 0.0 and gen.get("do_sample") is False,
              f"temperature {gen.get('temperature')}, do_sample {gen.get('do_sample')}"),
        check("chat template", chat, "applied (requests are chat messages)" if chat
              else "NOT applied: --apply_chat_template missing"),
        check("score", 0.176 <= score <= 0.324, f"{score:.3f} (paper 0.25, match range 0.176-0.324)"),
    ])

    ours = {row["id"]: row for row in map(json.loads, REFERENCE.read_text(encoding="utf8").splitlines())}
    same_answer = same_score = compared = 0
    for s in rows:
        ref = ours.get(str(s["doc"]["id"]))
        if ref is None:
            continue
        compared += 1
        theirs = s["filtered_resps"][0] if s["filtered_resps"] else None
        same_answer += (theirs or "").strip().lower() == (ref["action"] or "None").strip().lower()
        same_score += bool(s["score"]) == ref["correct"]
    print(f"\nPer-question comparison with our run ({compared} questions):")
    print(f"  same action: {same_answer}/{compared}   same correct/incorrect: {same_score}/{compared}")
    print(f"  ours {sum(r['correct'] for r in ours.values()) / len(ours):.3f}  vs  this run {score:.3f}")
    print("\nAll checks passed." if ok else "\nSome checks failed (see !! above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/official_reproduction"))
