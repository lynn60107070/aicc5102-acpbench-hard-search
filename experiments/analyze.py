
"""Analyze saved ACPBench-Hard experiment results.

Uses existing JSONL results. Does not make model API calls.
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import validator
from common.data import load_docs


def load_rows(path):
    """Read saved experiment results."""
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_documents():
    """Index official test documents by question ID."""
    return {str(doc["id"]): doc for doc in load_docs(split="test")}


def accuracy(rows):
    """Calculate the proportion of correct final answers."""
    if not rows:
        return 0.0
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def candidate_coverage(rows, documents):
    """Fraction of questions with a correct root candidate proposed."""
    proposed_correct = 0

    for row in rows:
        doc = documents[str(row["id"])]

        first_proposal = next(
            (reply["text"] for reply in row.get("replies", [])
             if reply["tag"] == "propose"),
            None,
        )

        if first_proposal is None:
            continue

        candidates = validator.parse_actions(first_proposal)

        if any(validator.score(doc, action).correct for action in candidates):
            proposed_correct += 1

    return proposed_correct / len(rows) if rows else 0.0

def extract_numeric_score(text):
    """Extract the final numeric Score: N value, if present."""
    matches = re.findall(r"\bScore\s*:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(matches[-1]) if matches else None


def score_separation(rows, documents):
    """Calculate candidate-score AUC and mean scores for correct/wrong candidates.

    Each proposal round is aligned with its following predictions.
    Only predictions with a matching numeric score are included.
    """
    scores = []

    for row in rows:
        doc = documents[str(row["id"])]
        replies = row.get("replies", [])
        first_proposal_index = next(
            (i for i, reply in enumerate(replies)
             if reply["tag"] == "propose"),
            None,
        )

        if first_proposal_index is None:
            continue

        candidates = validator.parse_actions(
            replies[first_proposal_index]["text"]
        )
        candidate_index = 0
        i = first_proposal_index + 1

        while i < len(replies) and replies[i]["tag"] != "propose":
            reply = replies[i]

            if reply["tag"] == "predict":
                prediction = reply["text"]
                if "NOT APPLICABLE" in prediction.upper():
                    candidate_index += 1
                    i += 1
                    continue

                score_text = None
                if i + 1 < len(replies) and replies[i + 1]["tag"] == "score":
                    score_text = replies[i + 1]["text"]

                if candidate_index < len(candidates) and score_text:
                    numeric_score = extract_numeric_score(score_text)
                    if numeric_score is not None:
                        verdict = validator.score(
                            doc, candidates[candidate_index]
                        )
                        scores.append((numeric_score, verdict.correct))

                candidate_index += 1

            i += 1

    correct_scores = [score for score, correct in scores if correct]
    wrong_scores = [score for score, correct in scores if not correct]

    if not correct_scores or not wrong_scores:
        auc = None
    else:
        wins = sum(
            1.0 if correct > wrong else 0.5 if correct == wrong else 0.0
            for correct in correct_scores
            for wrong in wrong_scores
        )
        auc = wins / (len(correct_scores) * len(wrong_scores))

    return {
        "auc": auc,
        "mean_correct_score": (
            statistics.mean(correct_scores) if correct_scores else None
        ),
        "mean_wrong_score": (
            statistics.mean(wrong_scores) if wrong_scores else None
        ),
        "n_correct_candidates": len(correct_scores),
        "n_wrong_candidates": len(wrong_scores),
    }

def summarize_run(rows, documents):
    """Summarize accuracy, errors, cost, candidate coverage, and score separation."""
    domains = defaultdict(list)
    errors = Counter()
    total_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for row in rows:
        domains[row["domain"]].append(bool(row["correct"]))
        if row.get("error"):
            errors[row["error"]] += 1

        total_calls += row.get("calls", 0)
        total_prompt_tokens += row.get("prompt_tokens", 0)
        total_completion_tokens += row.get("completion_tokens", 0)

    return {
        "n": len(rows),
        "accuracy": accuracy(rows),
        "accuracy_by_domain": {
            domain: sum(values) / len(values)
            for domain, values in sorted(domains.items())
        },
        "errors": dict(sorted(errors.items())),
        "total_calls": total_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "mean_calls": total_calls / len(rows) if rows else 0,
        "mean_tokens": (
            (total_prompt_tokens + total_completion_tokens) / len(rows)
            if rows else 0
        ),
        "candidate_coverage": candidate_coverage(rows, documents),
        "score_separation": score_separation(rows, documents),
    }


def aggregate_seed_metrics(summaries):
    """Aggregate run metrics by model size across seeds."""
    groups = defaultdict(list)

    for name, summary in summaries.items():
        match = re.search(r"main_search_(8b|70b)_s\d+", name)
        if match:
            groups[match.group(1)].append(summary)

    aggregated = {}
    for model, runs in sorted(groups.items()):
        metrics = {}
        for key in ("accuracy", "candidate_coverage"):
            values = [run[key] for run in runs]
            metrics[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        aucs = [
            run["score_separation"]["auc"]
            for run in runs
            if run["score_separation"]["auc"] is not None
        ]
        metrics["auc"] = {
            "mean": statistics.mean(aucs) if aucs else None,
            "std": statistics.stdev(aucs) if len(aucs) > 1 else 0.0,
        }
        metrics["total_calls_mean"] = statistics.mean(
            run["total_calls"] for run in runs
        )
        metrics["total_tokens_mean"] = statistics.mean(
            run["total_tokens"] for run in runs
        )
        aggregated[model] = metrics

    return aggregated


def write_latex_tables(summaries, aggregated, output_dir):
    """Write cross-seed and per-domain LaTeX tables."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Search performance across random seeds.}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Model & Accuracy & Candidate coverage & Score AUC \\",
        r"\hline",
    ]

    for model, metrics in aggregated.items():
        acc = metrics["accuracy"]
        cov = metrics["candidate_coverage"]
        auc = metrics["auc"]
        auc_text = "--" if auc["mean"] is None else f'{auc["mean"]:.3f}'
        lines.append(
            f'{model.upper()} & '
            f'{acc["mean"]:.3f} $\\pm$ {acc["std"]:.3f} & '
            f'{cov["mean"]:.3f} $\\pm$ {cov["std"]:.3f} & '
            f'{auc_text} \\\\'
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (output_dir / "cross_seed_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    domain_values = defaultdict(lambda: defaultdict(list))
    for name, summary in summaries.items():
        match = re.search(r"main_search_(8b|70b)_s\d+", name)
        if not match:
            continue
        model = match.group(1)
        for domain, value in summary["accuracy_by_domain"].items():
            domain_values[domain][model].append(value)

    domain_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Accuracy by domain, averaged across seeds.}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Domain & 8B accuracy & 70B accuracy \\",
        r"\hline",
    ]

    for domain in sorted(domain_values):
        values = domain_values[domain]
        a = statistics.mean(values["8b"]) if values["8b"] else float("nan")
        b = statistics.mean(values["70b"]) if values["70b"] else float("nan")
        domain_lines.append(f"{domain} & {a:.3f} & {b:.3f} \\\\")

    domain_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (output_dir / "domain_table.tex").write_text(
        "\n".join(domain_lines), encoding="utf-8"
    )
    # Compare saved direct baselines with mean search-run costs.
    direct_runs = {
        "8B direct (CoreWeave BF16)": "repro_llama31_8b_direct_bf16_coreweave",
        "8B direct (mixed providers)": "repro_llama31_8b_direct_mixed_providers",
        "70B direct (Bedrock)": "repro_llama31_70b_direct_bedrock",
    }

    cost_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Calls and token usage per 130 questions. Search values are means across three seeds; direct values are single saved runs.}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Method & Total calls & Total tokens \\",
        r"\hline",
    ]

    for label, run_name in direct_runs.items():
        direct_path = Path("results/final") / run_name / "summary.json"
        if direct_path.exists():
            direct = json.loads(direct_path.read_text(encoding="utf-8"))
            cost_lines.append(
                f"{label} & {direct['total_calls']} & {direct['total_tokens']} \\\\"
            )

    for model, metrics in aggregated.items():
        cost_lines.append(
            f"{model.upper()} search (mean) & "
            f"{metrics['total_calls_mean']:.1f} & "
            f"{metrics['total_tokens_mean']:.0f} \\\\"
        )

    cost_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (output_dir / "cost_table.tex").write_text(
        "\n".join(cost_lines), encoding="utf-8"
    )
    # Summarize evaluator error types across search seeds.
    error_counts = {
        "8b": defaultdict(list),
        "70b": defaultdict(list),
    }

    for name, summary in summaries.items():
        match = re.search(r"main_search_(8b|70b)_s\d+", name)
        if not match:
            continue

        model = match.group(1)
        for error_type, count in summary.get("errors", {}).items():
            error_counts[model][error_type].append(count)

    error_types = sorted({
        error_type
        for model_counts in error_counts.values()
        for error_type in model_counts
    })

    error_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Mean evaluator error counts per 130 questions across three search seeds.}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Error type & 8B mean & 70B mean \\",
        r"\hline",
    ]

    for error_type in error_types:
        means = []
        for model in ("8b", "70b"):
            counts = error_counts[model].get(error_type, [])
            means.append(
                f"{statistics.mean(counts) if counts else 0:.2f}"
            )

        escaped_type = error_type.replace("_", r"\_")
        error_lines.append(
            f"{escaped_type} & {means[0]} & {means[1]} " + r"\\"
        )

    error_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (output_dir / "error_table.tex").write_text(
        "\n".join(error_lines) + "\n", encoding="utf-8"
    )


def write_figures(aggregated, output_dir):
    """Create accuracy and AUC comparison figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping figures because matplotlib is unavailable: {exc}")
        return

    models = list(aggregated)
    labels = [model.upper() for model in models]

    for metric, title, filename in [
        ("accuracy", "Search accuracy across seeds", "accuracy.png"),
        ("auc", "Candidate score-separation AUC", "score_auc.png"),
    ]:
        means = []
        errors = []
        for model in models:
            item = aggregated[model][metric]
            means.append(item["mean"] if item["mean"] is not None else 0.0)
            errors.append(item["std"])

        fig, ax = plt.subplots()
        ax.bar(labels, means, yerr=errors, capsize=5)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze saved ACPBench-Hard experiment results."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run folders containing rows.jsonl",
    )
    parser.add_argument(
        "--out",
        default="results/analysis",
        help="Directory for analysis outputs",
    )
    args = parser.parse_args()

    documents = load_documents()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}

    for run_path in args.runs:
        path = Path(run_path)
        rows_path = path / "rows.jsonl" if path.is_dir() else path
        run_name = path.name if path.is_dir() else path.parent.name

        rows = load_rows(rows_path)
        missing = [str(row["id"]) for row in rows
                   if str(row["id"]) not in documents]
        if missing:
            raise ValueError(
                f"{run_name}: {len(missing)} question IDs missing from dataset"
            )

        summaries[run_name] = summarize_run(rows, documents)
        print(
            f"{run_name}: accuracy={summaries[run_name]['accuracy']:.3f}, "
            f"coverage={summaries[run_name]['candidate_coverage']:.3f}, "
            f"AUC={summaries[run_name]['score_separation']['auc']}"
        )

    report = {
        "runs": summaries,
        "note": "Computed from saved result rows; no model calls performed.",
    }
    aggregated = aggregate_seed_metrics(summaries)
    report["aggregate_by_model"] = aggregated
    write_latex_tables(summaries, aggregated, output_dir)
    write_figures(aggregated, output_dir)
    print("Saved LaTeX tables and figures.")
    report_path = output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
