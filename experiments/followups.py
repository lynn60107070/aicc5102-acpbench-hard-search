"""Post-hoc follow-up table (Table 6) and the extra statistics in Appendix D.

Reads the saved runs in results/final/ and makes no model calls:

    python -m experiments.followups

Writes paper/tables/followups.tex (Table 7), paper/tables/validation_search.tex, paper/figures/vote_curve.pdf and results/analysis/followups.json, and prints:
- per-seed exact McNemar p-values for the follow-up comparisons;
- seed-pooled tests: for each question, average correctness over the seeds, then a paired
  sign-flip permutation test and a bootstrap 95% interval on the mean difference;
- a one-sample binomial test of each structured+vote seed against the paper's 0.25;
- how far the 14,000-token cap was exceeded;
- prompt vs. generated tokens per question;
- structured+vote accuracy when voting over only its first m samples.
"""

import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.validator import parse_action
from experiments.compare import compare_correctness, match_questions, mcnemar_exact

ROOT = Path("results/final")
SEEDS = (1, 2, 3)
CAP = 14000

RUNS = {
    "direct": ["repro_llama31_8b_direct_bf16_coreweave"],
    "structured": ["structured_8b"],
    "bon14": [f"best_of_n14_8b_s{s}" for s in SEEDS],
    "search": [f"main_search_8b_s{s}" for s in SEEDS],
    "sv": [f"followup_structured_vote/followup_sv_8b_s{s}" for s in SEEDS],
    "bon_cap": [f"a7/a7_bon_s{s}" for s in SEEDS],
    "search_cap": [f"a7/a7_search_s{s}" for s in SEEDS],
    "sv_cap": [f"followup_structured_vote/followup_sv14k_8b_s{s}" for s in SEEDS],
    "fixed": [f"ablations/abl8b_fixed_s{s}" for s in SEEDS],
    "k5": [f"ablations/abl8b_k5_s{s}" for s in SEEDS],
    "direct70": ["repro_llama31_70b_direct_bedrock"],
    "search70": [f"main_search_70b_s{s}" for s in SEEDS],
    # 70B controls and follow-up, pre-registered in experiments/preregistration_22nov.md
    "structured70": ["structured_70b"],
    "bon70": [f"best_of_n14_70b_s{s}" for s in SEEDS],
    "sv70": [f"followup_structured_vote/followup_sv_70b_s{s}" for s in SEEDS],
}
OPTIONAL = {"structured70", "bon70", "sv70"}  # skipped (with a note) until their runs exist

TABLE_ROWS = [  # (label, key); None draws a rule
    ("Direct generation", "direct"),
    ("Structured prompting", "structured"),
    ("Best-of-N ($N{=}14$)", "bon14"),
    ("Search", "search"),
    (r"\textbf{Structured + vote} ($N{=}14$)", "sv"),
    None,
    ("Best-of-N ($N{=}8$), 14k-token cap", "bon_cap"),
    ("Search, 14k-token cap", "search_cap"),
    ("Structured + vote, 14k-token cap", "sv_cap"),
    "70B",
    ("Direct generation", "direct70"),
    ("Structured prompting", "structured70"),
    ("Best-of-N ($N{=}14$)", "bon70"),
    ("Search", "search70"),
    (r"\textbf{Structured + vote} ($N{=}14$)", "sv70"),
]

ABLATIONS = ["b28", "b7", "d1", "d3", "fewshot", "fixed", "k2", "k5", "nopredict", "noprune",
             "random", "steps", "vote"]

PER_SEED = [  # (A, B): is A different from B, seed by seed
    ("sv", "direct"), ("sv", "bon14"), ("sv", "structured"), ("sv_cap", "bon_cap"),
    ("sv_cap", "direct"), ("search_cap", "direct"), ("k5", "search"),
    ("sv70", "bon70"), ("sv70", "direct70"), ("bon70", "direct70"), ("structured70", "direct70"),
]

POOLED = [
    ("bon14", "search"), ("search", "direct"), ("bon14", "direct"), ("search70", "direct70"),
    ("sv", "direct"), ("sv", "bon14"), ("sv", "structured"), ("bon_cap", "search_cap"),
    ("sv_cap", "bon_cap"), ("fixed", "search"),
    ("sv70", "bon70"), ("sv70", "direct70"),  # pre-registered 22 Nov
]


def load(name):
    with open(ROOT / name / "rows.jsonl", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def acc(rows):
    return sum(bool(r["correct"]) for r in rows) / len(rows)


def tokens(r):
    return r["prompt_tokens"] + r["completion_tokens"]


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def per_question_mean(runs):
    by_id = {}
    for rows in runs:
        for r in rows:
            by_id.setdefault(str(r["id"]), []).append(bool(r["correct"]))
    return {q: statistics.mean(v) for q, v in by_id.items()}


def pooled_test(runs_a, runs_b, n_perm=20000, n_boot=10000, seed=0):
    """Paired test on per-question seed-averaged correctness (a single-seed run is used as is)."""
    a, b = per_question_mean(runs_a), per_question_mean(runs_b)
    ids = sorted(a)
    assert sorted(b) == ids
    diffs = [a[q] - b[q] for q in ids]
    observed = statistics.mean(diffs)
    rng = random.Random(seed)
    extreme = sum(
        abs(statistics.mean(d if rng.random() < 0.5 else -d for d in diffs)) >= abs(observed) - 1e-12
        for _ in range(n_perm)
    )
    boots = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(n_boot))
    return {"diff": observed, "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
            "p": extreme / n_perm}


def binomial_two_sided(k, n, p0):
    probs = [math.comb(n, j) * p0 ** j * (1 - p0) ** (n - j) for j in range(n + 1)]
    return sum(p for p in probs if p <= probs[k] * (1 + 1e-9))


def structured_action(text):
    """The final action of one structured reply, parsed as methods/structured_vote.py does."""
    m = list(re.finditer(r"final action[\s:*]*", text, flags=re.IGNORECASE))
    return parse_action(text[m[-1].end():] if m else text)


def domain_cluster_bootstrap(runs_a, runs_b, n_boot=10000, seed=0):
    """95% interval for the mean paired difference, resampling the 13 domains (questions within a
    domain move together), plus the share of resamples on the other side of zero (x2)."""
    a, b = per_question_mean(runs_a), per_question_mean(runs_b)
    domain = {str(r["id"]): r["domain"] for r in runs_a[0]}
    by_domain = {}
    for q in a:
        by_domain.setdefault(domain[q], []).append(a[q] - b[q])
    groups = list(by_domain.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [d for _ in groups for d in rng.choice(groups)]
        boots.append(statistics.mean(sample))
    boots.sort()
    observed = statistics.mean(d for g in groups for d in g)
    other_side = sum((x <= 0) if observed > 0 else (x >= 0) for x in boots) / n_boot
    return {"diff": observed, "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
            "p_approx": min(1.0, 2 * other_side), "domains": len(groups)}


def validation_search_table():
    """Re-created validation comparison of the evaluators (pre-registered 22 Nov), if present."""
    root = ROOT / "validation_search"
    if not root.exists():
        return None
    names = {"llm": "0--10 score (frozen)", "steps": "Steps to goal", "vote": "Vote ($d{=}1$)",
             "fixed": "Proposer's order (fixed)", "random": "Random scores"}
    out, lines = {}, [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Search evaluators on the 40 validation questions (8B), re-created on 22 Nov with the final "
        r"frozen code (pre-registered; the original 23 Oct runs were not kept). Correct answers out of 40 "
        r"for seeds 1, 2, 3. Generated by \texttt{experiments/followups.py}.}",
        r"\label{tab:validation-search}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
        r"Evaluator & Correct (seeds 1, 2, 3) & Mean \\", r"\midrule"]
    for key, label in names.items():
        counts = []
        for seed in SEEDS:
            path = root / f"val_search_{key}_s{seed}" / "rows.jsonl"
            if not path.exists():
                return None
            run = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            counts.append(sum(bool(r["correct"]) for r in run))
        out[key] = counts
        lines.append(f"{label} & {', '.join(map(str, counts))} & {statistics.mean(counts):.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path("paper/tables/validation_search.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def vote_figure(report):
    """Appendix figure: structured + vote accuracy against the number of samples voted over."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    curves = [("sv_first_m", "Structured + vote, 8B", "#3b6a9b", "o")]
    if "sv70_first_m" in report:
        curves.append(("sv70_first_m", "Structured + vote, 70B", "#1f2933", "s"))
    for key, label, color, marker in curves:
        ms = sorted(int(m) for m in report[key])
        ax.plot(ms, [report[key][str(m)]["accuracy"] if str(m) in report[key] else report[key][m]["accuracy"]
                     for m in ms], marker=marker, color=color, label=label, lw=1.6, ms=4)
    t = report["table"]
    refs = [("direct", "Direct, 8B", "#3b6a9b", ":"), ("bon14", "Best-of-N (14 calls), 8B", "#3b6a9b", "--"),
            ("search", "Search, 8B", "#3b6a9b", "-.")]
    if "bon70" in t:
        refs += [("structured70", "Greedy structured (1 call), 70B", "#1f2933", (0, (1, 1))),
                 ("direct70", "Direct, 70B", "#1f2933", ":"), ("bon70", "Best-of-N (14 calls), 70B", "#1f2933", "--"),
                 ("search70", "Search, 70B", "#1f2933", "-.")]
    for key, label, color, style in refs:
        ax.axhline(t[key]["accuracy"], color=color, ls=style, lw=1, alpha=0.75, label=label)
    ax.set_xlabel("Samples voted over (calls per question)")
    ax.set_ylabel("Accuracy (test, mean of 3 seeds)")
    ax.set_xlim(0.5, 14.5)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14])
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("paper/figures/vote_curve.pdf")
    plt.close(fig)


def main():
    rows = {}
    for key, names in RUNS.items():
        if key in OPTIONAL and not all((ROOT / n / "rows.jsonl").exists() for n in names):
            print(f"skipping {key}: runs not in results/final yet")
            continue
        rows[key] = [load(name) for name in names]
    report = {"note": "Computed from results/final/; no model calls."}

    # ---- Table 6
    table = {}
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Post-hoc follow-up on the test split (130 questions, seeds 1--3 for sampled methods). The 70B",
        r"structured, best-of-$N$ and structured + vote runs were pre-registered on 22 Nov; 70B direct and search are the main runs. Generated by \texttt{experiments/followups.py}.}",
        r"\label{tab:followups}",
        r"\small",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Method & Accuracy & Calls & Tokens \\",
        r"\midrule",
    ]
    lines.append(r"\multicolumn{4}{@{}l}{\emph{Llama 3.1 8B}} \\")
    for item in TABLE_ROWS:
        if item is None:
            lines.append(r"\midrule")
            continue
        if item == "70B":
            if "sv70" not in rows:
                break
            lines += [r"\midrule", r"\multicolumn{4}{@{}l}{\emph{Llama 3.1 70B}} \\"]
            continue
        label, key = item
        runs = rows[key]
        m, s = mean_std([acc(r) for r in runs])
        calls = statistics.mean(sum(x["calls"] for x in r) / len(r) for r in runs)
        toks = statistics.mean(sum(tokens(x) for x in r) / len(r) for r in runs)
        table[key] = {"accuracy": m, "std": s, "per_seed": [acc(r) for r in runs],
                      "calls": calls, "tokens": toks}
        acc_text = f"{m:.3f}" if len(runs) == 1 else f"{m:.3f} $\\pm$ {s:.3f}"
        if key in ("sv", "sv70"):
            acc_text = r"\textbf{" + acc_text + "}"
        lines.append(f"{label} & {acc_text} & {calls:.1f} & {toks:,.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path("paper/tables/followups.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["table"] = table

    # ---- per-seed McNemar
    report["mcnemar_per_seed"] = {}
    for a, b in PER_SEED:
        if a not in rows or b not in rows:
            continue
        ps, diffs = [], []
        for i in range(max(len(rows[a]), len(rows[b]))):  # one comparison if both are single runs
            ra = rows[a][i] if len(rows[a]) > 1 else rows[a][0]
            rb = rows[b][i] if len(rows[b]) > 1 else rows[b][0]
            ma, mb = match_questions(ra, rb)
            _, a_only, b_only, _ = compare_correctness(ma, mb)
            ps.append(mcnemar_exact(a_only, b_only))
            diffs.append(sum(x["correct"] for x in ra) - sum(x["correct"] for x in rb))
        report["mcnemar_per_seed"][f"{a} vs {b}"] = {"p": ps, "question_diff": diffs}

    # ---- every 8B ablation against the frozen search, seed by seed (Discussion: only k=5, seed 2)
    report["ablations_vs_search"] = {}
    for name in ABLATIONS:
        ps, diffs = [], []
        for s, frozen in zip(SEEDS, rows["search"]):
            abl = load(f"ablations/abl8b_{name}_s{s}")
            ma, mb = match_questions(abl, frozen)
            _, a_only, b_only, _ = compare_correctness(ma, mb)
            ps.append(mcnemar_exact(a_only, b_only))
            diffs.append(sum(x["correct"] for x in abl) - sum(x["correct"] for x in frozen))
        report["ablations_vs_search"][name] = {"p": ps, "question_diff": diffs}

    # ---- seed-pooled tests
    report["pooled"] = {f"{a} vs {b}": pooled_test(rows[a], rows[b]) for a, b in POOLED
                        if a in rows and b in rows}
    # Reported in Appendix D but not pre-registered, so kept out of the Holm family above.
    report["extra_pooled"] = {f"{a} vs {b}": pooled_test(rows[a], rows[b])
                              for a, b in [("sv70", "structured70")] if a in rows and b in rows}
    report["domain_cluster"] = {f"{a} vs {b}": domain_cluster_bootstrap(rows[a], rows[b])
                                for a, b in [("sv", "bon14"), ("sv_cap", "bon_cap"), ("sv", "direct"), ("bon14", "search"),
                                             ("sv70", "bon70"), ("sv70", "direct70")]
                                if a in rows and b in rows}
    # Holm correction over the pooled family. The sign-flip test is approximate when a 3-seed
    # mean is compared with a single greedy run, since their spreads differ under the null.
    names = sorted(report["pooled"], key=lambda k: report["pooled"][k]["p"])
    running = 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * report["pooled"][name]["p"]))
        report["pooled"][name]["p_holm"] = running

    # ---- structured+vote against the paper's single-call 0.25
    report["sv_vs_paper_0.25"] = [
        binomial_two_sided(sum(x["correct"] for x in r), len(r), 0.25) for r in rows["sv"]]
    if "sv70" in rows:
        report["sv70_vs_paper_0.42"] = [
            binomial_two_sided(sum(x["correct"] for x in r), len(r), 0.42) for r in rows["sv70"]]

    # ---- the 14,000-token cap is checked before each call, so runs can exceed it
    report["cap_overshoot"] = {
        key: {"over_cap": [sum(tokens(x) > CAP for x in r) for r in rows[key]],
              "mean": statistics.mean(tokens(x) for r in rows[key] for x in r),
              "max": max(tokens(x) for r in rows[key] for x in r)}
        for key in ("bon_cap", "search_cap", "sv_cap")}

    # ---- prompt vs generated tokens
    report["token_split"] = {
        key: {"prompt": statistics.mean(x["prompt_tokens"] for r in rows[key] for x in r),
              "completion": statistics.mean(x["completion_tokens"] for r in rows[key] for x in r)}
        for key in ("direct", "bon14", "search", "sv", "search70")}

    # ---- structured+vote over its first m samples. Capped runs replay the same cached
    # samples, so this is what a cap at m calls gives.
    for r_full, r_cap in zip(rows["sv"], rows["sv_cap"]):
        full_by_id = {str(x["id"]): x for x in r_full}
        for b in r_cap:
            a, k = full_by_id[str(b["id"])], len(b["replies"])
            assert [x["text"] for x in a["replies"][:k]] == [x["text"] for x in b["replies"]]
    per_call = statistics.mean(tokens(x) / x["calls"] for r in rows["sv"] for x in r)
    report["sv_first_m"] = vote_curve(rows["sv"], per_call)
    report["sv_token_matched"] = token_matched_vote(rows["sv"], rows["bon_cap"])
    if "sv70" in rows:
        per_call70 = statistics.mean(tokens(x) / x["calls"] for r in rows["sv70"] for x in r)
        report["sv70_first_m"] = vote_curve(rows["sv70"], per_call70)
    report["validation_search"] = validation_search_table()
    vote_figure(report)

    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    Path("results/analysis/followups.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def token_matched_vote(sv_runs, bon_cap_runs):
    """Structured+vote with a per-question budget fixed in advance (no choice of m from the curve).

    "bon": each question gets as many samples as fit in the tokens capped best-of-N used on that
    question (same seed). "cap": as many as fit under 14,000 tokens, never crossing it. Tokens per
    sample are the question's own average (row tokens / calls); at least one sample is kept.
    """
    from common import validator
    from common.data import load_docs
    docs = {str(d["id"]): d for d in load_docs(split="test")}
    out = {}
    for rule in ("bon", "cap"):
        accs, samples, toks = [], [], []
        for sv, bon in zip(sv_runs, bon_cap_runs):
            bon_tokens = {str(x["id"]): tokens(x) for x in bon}
            correct = 0
            for x in sv:
                per_sample = tokens(x) / x["calls"]
                budget = bon_tokens[str(x["id"])] if rule == "bon" else CAP
                m = max(1, min(len(x["replies"]), int(budget // per_sample)))
                samples.append(m)
                toks.append(m * per_sample)
                actions = [a for a in (structured_action(y["text"]) for y in x["replies"][:m]) if a]
                if actions:
                    vote = Counter(actions).most_common(1)[0][0]
                    correct += validator.score(docs[str(x["id"])], vote).correct
            accs.append(correct / len(sv))
        out[rule] = {"accuracy": statistics.mean(accs), "per_seed": accs,
                     "mean_samples": statistics.mean(samples), "mean_tokens": statistics.mean(toks)}
    return out


def vote_curve(runs, per_call):
    """Accuracy of voting over the first m samples, scored with the official validator."""
    from common import validator
    from common.data import load_docs
    docs = {str(d["id"]): d for d in load_docs(split="test")}
    cache = {}
    out = {}
    for m in range(1, 15):
        accs = []
        for run in runs:
            correct = 0
            for x in run:
                actions = [a for a in (structured_action(y["text"]) for y in x["replies"][:m]) if a]
                if not actions:
                    continue
                vote = Counter(actions).most_common(1)[0][0]
                key = (str(x["id"]), vote)
                if key not in cache:
                    cache[key] = validator.score(docs[str(x["id"])], vote).correct
                correct += cache[key]
            accs.append(correct / len(run))
        out[m] = {"accuracy": statistics.mean(accs), "per_seed": accs,
                  "approx_tokens": round(m * per_call)}
    return out


if __name__ == "__main__":
    main()
