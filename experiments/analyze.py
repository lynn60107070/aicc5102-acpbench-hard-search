
"""Analyze saved ACPBench-Hard experiment results.

Uses existing JSONL results. Does not make model API calls.
"""

import argparse
import glob
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import validator
from common.data import load_docs
from experiments.compare import match_questions, compare_correctness, mcnemar_exact


def load_rows(path):
    with open(path, encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_documents():
    """Index official test documents by question ID."""
    return {str(doc["id"]): doc for doc in load_docs(split="test")}


def accuracy(rows):
    """Calculate the proportion of correct final answers."""
    if not rows:
        return 0.0
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def binomial_interval_95(paper_accuracy, n=130):
    """Return the 95% normal-approximation interval around a reported accuracy."""
    standard_error = (paper_accuracy * (1 - paper_accuracy) / n) ** 0.5
    return (paper_accuracy - 1.96 * standard_error,
            paper_accuracy + 1.96 * standard_error)


def within_95_binomial_interval(paper_accuracy, reproduced_accuracy, n=130):
    """Check whether reproduced accuracy is inside the paper's 95% binomial interval."""
    standard_error = (paper_accuracy * (1 - paper_accuracy) / n) ** 0.5
    lower = paper_accuracy - 1.96 * standard_error
    upper = paper_accuracy + 1.96 * standard_error
    return lower <= reproduced_accuracy <= upper


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

def calculate_auc(scores_and_labels):
    """Calculate pairwise AUC from (score, is_correct) pairs.

    Returns None if either class is absent.
    """
    correct_scores = [
        score for score, correct in scores_and_labels if correct
    ]
    wrong_scores = [
        score for score, correct in scores_and_labels if not correct
    ]

    if not correct_scores or not wrong_scores:
        return None

    wins = sum(
        1.0 if correct > wrong else 0.5 if correct == wrong else 0.0
        for correct in correct_scores
        for wrong in wrong_scores
    )
    return wins / (len(correct_scores) * len(wrong_scores))

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

    auc = calculate_auc(scores)

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

def compare_direct_to_search(direct_rows, search_rows):
    """Compare one direct baseline against one search seed."""
    run_direct, run_search = match_questions(direct_rows, search_rows)

    _, direct_only, search_only, _ = compare_correctness(
        run_direct, run_search
    )

    return {
        "direct_accuracy": accuracy(direct_rows),
        "search_accuracy": accuracy(search_rows),
        "p_value": mcnemar_exact(direct_only, search_only),
        "n_questions": len(direct_rows),
    }
def aggregate_ablation_runs(run_rows):
    """Aggregate accuracy and calls across seeds for each ablation setting."""
    groups = defaultdict(list)

    for name, rows in run_rows.items():
        match = re.fullmatch(r"(abl8b_.+)_s(\d+)", name)
        if match:
            groups[match.group(1)].append(rows)

    aggregated = {}
    for setting, runs in sorted(groups.items()):
        accuracies = [accuracy(rows) for rows in runs]
        calls_per_question = [
            sum(row.get("calls", 0) for row in rows) / len(rows)
            if rows else 0.0
            for rows in runs
        ]

        aggregated[setting] = {
            "n_seeds": len(runs),
            "accuracy_mean": statistics.mean(accuracies),
            "accuracy_std": (
                statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
            ),
            "calls_mean": statistics.mean(calls_per_question),
            "calls_std": (
                statistics.stdev(calls_per_question)
                if len(calls_per_question) > 1 else 0.0
            ),
        }

    return aggregated

def write_latex_tables(summaries, aggregated, output_dir):
    """Write cross-seed and per-domain LaTeX tables."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Search performance across random seeds.}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Accuracy & Candidate coverage & Score AUC \\",
        r"\midrule",
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
        r"\bottomrule",
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
        r"\caption{Search accuracy by domain on the 130 test questions (10 per domain), mean across seeds 1--3.}",
        r"\label{tab:domains}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Domain & 8B accuracy & 70B accuracy \\",
        r"\midrule",
    ]

    for domain in sorted(domain_values):
        values = domain_values[domain]
        a = statistics.mean(values["8b"]) if values["8b"] else float("nan")
        b = statistics.mean(values["70b"]) if values["70b"] else float("nan")
        domain_lines.append(f"{domain} & {a:.3f} & {b:.3f} \\\\")

    domain_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
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
        r"\toprule",
        r"Method & Total calls & Total tokens \\",
        r"\midrule",
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
    for label, run_names in {
        "8B Structured": ["structured_8b"],
        "8B Best-of-N (mean)": [
            "best_of_n14_8b_s1",
            "best_of_n14_8b_s2",
            "best_of_n14_8b_s3",
        ],
    }.items():
        run_costs = []
        for run_name in run_names:
            summary_path = Path("results/final") / run_name / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                run_costs.append((
                    summary["total_calls"],
                    summary["total_tokens"],
                ))

        if run_costs:
            mean_calls = statistics.mean(calls for calls, _ in run_costs)
            mean_tokens = statistics.mean(tokens for _, tokens in run_costs)
            cost_lines.append(
                f"{label} & {mean_calls:.1f} & {mean_tokens:.0f} " + r"\\"
            )
    cost_lines.extend([
        r"\bottomrule",
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
        for error_type in ("format", "inapplicable", "not_closer"):
            error_counts[model][error_type].append(
                summary.get("errors", {}).get(error_type, 0)
            )

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
        r"\toprule",
        r"Error type & 8B mean & 70B mean \\",
        r"\midrule",
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
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (output_dir / "error_table.tex").write_text(
        "\n".join(error_lines) + "\n", encoding="utf-8"
    )


def write_main_results_table(run_rows, output_dir):
    """Write direct, structured, best-of-N, and search comparisons."""
    direct_runs = {
        "8b": "repro_llama31_8b_direct_bf16_coreweave",
        "70b": "repro_llama31_70b_direct_bedrock",
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Direct, structured, best-of-N, and search results. Controls were run for 8B only. Multi-seed methods report mean accuracy and sample standard deviation. Exact McNemar $p$-values compare each seed (1, 2, 3) with direct generation and are uncorrected. Calls and tokens are per question. The 14k-token-cap rows share a cap of 14,000 tokens per question.}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Method & Accuracy ($\pm$ std.) & Calls & Tokens & McNemar $p$ \\",
        r"\midrule",
    ]

    def cost(rows):
        calls = sum(row.get("calls", 0) for row in rows)
        tokens = sum(
            row.get("prompt_tokens", 0) + row.get("completion_tokens", 0)
            for row in rows
        )
        return calls, tokens

    def append_single(model, method, rows, direct_rows=None):
        if not rows:
            lines.append(
                f"{model.upper()} & {method} & pending & -- & -- & -- " + r"\\"
            )
            return

        calls, tokens = cost(rows)
        calls /= len(rows)
        tokens /= len(rows)
        p_text = "--"
        if direct_rows:
            p_text = f"{compare_direct_to_search(direct_rows, rows)['p_value']:.3f}"

        lines.append(
            f"{model.upper()} & {method} & {accuracy(rows):.3f} "
            f"& {calls:.1f} & {tokens:,.0f} & {p_text} " + r"\\"
        )

    def append_multiple(model, method, runs, direct_rows=None):
        if not runs:
            append_single(model, method, None)
            return

        accuracies = [accuracy(rows) for _, rows in runs]
        call_totals, token_totals = zip(*(
            (cost(rows)[0] / len(rows), cost(rows)[1] / len(rows))
            for _, rows in runs
        ))
        p_values = (
            [
                compare_direct_to_search(direct_rows, rows)["p_value"]
                for _, rows in runs
            ]
            if direct_rows else []
        )
        p_text = ", ".join(f"{p:.3f}" for p in p_values) if p_values else "--"

        mean_accuracy = statistics.mean(accuracies)
        std_accuracy = (
            statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        )
        lines.append(
            f"{model.upper()} & {method} & {mean_accuracy:.3f} "
            f"$\\pm$ {std_accuracy:.3f} "
            f"& {statistics.mean(call_totals):.1f} "
            f"& {statistics.mean(token_totals):,.0f} "
            f"& {p_text} " + r"\\"
        )

    for model in ("8b", "70b"):
        direct_rows = run_rows.get(direct_runs[model])
        append_single(model, "Direct", direct_rows)

        search_runs = sorted(
            [
                (name, rows)
                for name, rows in run_rows.items()
                if re.fullmatch(rf"main_search_{model}_s\d+", name)
            ],
            key=lambda item: item[0],
        )
        append_multiple(model, "Search", search_runs, direct_rows)

        if model == "8b":
            structured_rows = run_rows.get("structured_8b")
            append_single(model, "Structured", structured_rows, direct_rows)

            best_of_n_runs = sorted(
                [
                    (name, rows)
                    for name, rows in run_rows.items()
                    if re.fullmatch(r"best_of_n\d+_8b_s\d+", name)
                ],
                key=lambda item: item[0],
            )
            append_multiple(
                model, "Best-of-N ($N{=}14$)", best_of_n_runs, direct_rows
            )

            # Token-matched check: both methods share a 14,000-token cap.
            for label, pattern in (
                ("Search, 14k-token cap", r"a7_search_s\d+"),
                (r"Best-of-N ($N{=}8$), 14k-token cap", r"a7_bon_s\d+"),
            ):
                capped_runs = sorted(
                    [
                        (name, rows)
                        for name, rows in run_rows.items()
                        if re.fullmatch(pattern, name)
                    ],
                    key=lambda item: item[0],
                )
                if capped_runs:
                    append_multiple(model, label, capped_runs, direct_rows)

        if model == "8b":
            lines.append(r"\midrule")

    # Remove the final separator if one was added after the last model.
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "main_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def load_progression_score(run_name):
    """Read the saved aggregate score for a Progression reproduction run."""
    run_dir = Path("results/final") / run_name
    result_files = sorted(run_dir.glob("results_*.json"))

    if not result_files:
        return None

    result_data = json.loads(
        result_files[-1].read_text(encoding="utf-8")
    )
    task_result = result_data.get("results", {}).get("acp_prog_gen", {})
    return task_result.get("score,acp_grammar_parse")


def write_reproduction_table(run_rows, output_dir):
    """Write paper-versus-reproduced accuracy for the four reference runs."""
    reference_runs = [
        ("Llama 3.1 8B", "Next Action",
         "repro_llama31_8b_direct_bf16_coreweave", 0.250),
        ("Llama 3.1 70B", "Next Action",
         "repro_llama31_70b_direct_bedrock", 0.420),
        ("Llama 3.1 8B", "Progression",
         "repro_prog_llama31_8b", 0.400),
        ("Llama 3.1 70B", "Progression",
         "repro_prog_llama31_70b", 0.650),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Paper reference accuracies and reproduced results on ACPBench-Hard, with the 95\% binomial interval around each reported value ($n=130$).}",
        r"\label{tab:reproduction}",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Model & Task & Paper accuracy & Reproduced accuracy & 95\% interval \\",
        r"\midrule",
    ]

    for model, task, run_name, paper_accuracy in reference_runs:
        rows = run_rows.get(run_name)

        if task == "Progression":
            score = load_progression_score(run_name)
            reproduced_value = score
        else:
            reproduced_value = accuracy(rows) if rows else None

        reproduced = (
            f"{reproduced_value:.3f}"
            if reproduced_value is not None
            else "--"
        )

        lower, upper = binomial_interval_95(paper_accuracy)
        within_interval = (
            f"[{lower:.3f}, {upper:.3f}]" + r"\,\checkmark"
            if reproduced_value is not None
            and within_95_binomial_interval(paper_accuracy, reproduced_value)
            else "--"
        )

        lines.append(
            f"{model} & {task} & {paper_accuracy:.3f} "
            f"& {reproduced} & {within_interval} " + r"\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reproduction.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
def write_ablation_table(ablations, output_dir):
    """Write paper-ready ablation results using booktabs."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Search ablations on Llama 3.1 8B. Accuracy is mean $\pm$ standard deviation across seeds; calls are mean calls per question. Rows with identical accuracy reuse the same cached proposals and select the same answers (Section~\ref{sec:method}).}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Setting & Accuracy ($\pm$ std.) & Mean calls / question \\",
        r"\midrule",
        r"Frozen search (reference) & 0.159 $\pm$ 0.012 & 13.4 \\",
    ]

    group_order = [
        ("Scoring variants", ["fixed", "random", "steps", "vote"]),
        ("Tree shape ($k$, $d$)", ["k2", "k5", "d1", "d3"]),
        ("Budget", ["b7", "b28"]),
        ("Prompt", ["fewshot"]),
        ("Other search mechanics", ["nopredict", "noprune"]),
    ]

    for group_name, settings in group_order:
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{group_name}}}}} \\")
        for setting in settings:
            full_setting = f"abl8b_{setting}"
            if full_setting not in ablations:
                continue
            metrics = ablations[full_setting]
            label = setting.removeprefix("abl8b_")
            label_map = {
                "b7": "budget=7",
                "b28": "budget=28",
                "d1": "depth=1",
                "d3": "depth=3",
                "fewshot": "few-shot",
                "fixed": "scoring=fixed",
                "k2": "branching=2",
                "k5": "branching=5",
                "nopredict": "no prediction",
                "noprune": "no pruning",
                "random": "scoring=random",
                "steps": "scoring=steps",
                "vote": "scoring=vote",
            }
            label = label_map.get(label, label.replace("_", r"\_"))
            if metrics["n_seeds"] != 3:
                accuracy_text = (
                    f'{metrics["accuracy_mean"]:.3f} $\\pm$ '
                    f'{metrics["accuracy_std"]:.3f} '
                    f'({metrics["n_seeds"]} seeds)'
                )
            else:
                accuracy_text = (
                    f'{metrics["accuracy_mean"]:.3f} $\\pm$ '
                    f'{metrics["accuracy_std"]:.3f}'
                )

            lines.append(
                f'{label} & {accuracy_text} & {metrics["calls_mean"]:.1f} \\\\'
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (output_dir / "ablations.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

def write_figures(aggregated, output_dir):
    """Create accuracy and AUC comparison figures."""
    try:
        import matplotlib.pyplot as plt
        paper_figures_dir = Path("paper/figures")
        paper_figures_dir.mkdir(parents=True, exist_ok=True)
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
        fig.savefig(paper_figures_dir / f"{Path(filename).stem}.pdf")
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
    run_rows = {}
    summaries = {}

    # cmd.exe does not expand wildcards, so expand them here.
    run_paths = []
    for pattern in args.runs:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        run_paths.extend(matches)

    for run_path in run_paths:
        path = Path(run_path)
        rows_path = path / "rows.jsonl" if path.is_dir() else path
        run_name = path.name if path.is_dir() else path.parent.name

        # Progression runs contain aggregate results JSON, not rows.jsonl.
        if run_name.startswith("repro_prog_"):
            score = load_progression_score(run_name)
            if score is None:
                raise FileNotFoundError(
                    f"{run_name}: no aggregate Progression score found"
                )
            print(f"{run_name}: aggregate accuracy={score:.3f}")
            continue

        rows = load_rows(rows_path)
        run_rows[run_name] = rows
        missing = [
            str(row["id"]) for row in rows
            if str(row["id"]) not in documents
        ]
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
    ablations = aggregate_ablation_runs(run_rows)
    report["ablations"] = ablations
    paper_tables_dir = Path("paper/tables")
    write_latex_tables(summaries, aggregated, output_dir)
    (paper_tables_dir / "domain_table.tex").write_text(
        (output_dir / "domain_table.tex").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_ablation_table(ablations, paper_tables_dir)
    write_main_results_table(run_rows, paper_tables_dir)
    write_reproduction_table(run_rows, paper_tables_dir)
    write_figures(aggregated, output_dir)
    print("Saved LaTeX tables and figures.")
    report_path = output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
