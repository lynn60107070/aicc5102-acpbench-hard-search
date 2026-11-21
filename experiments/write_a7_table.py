import json
import statistics
from pathlib import Path

root = Path("results/final/a7")  # the saved runs; results/runs/ is not in the repo

groups = {
    "Search": [f"a7_search_s{i}" for i in (1, 2, 3)],
    r"Best-of-N ($N=8$)": [f"a7_bon_s{i}" for i in (1, 2, 3)],
}


def mean_std(summaries, key):
    values = [s[key] for s in summaries]
    return statistics.mean(values), statistics.stdev(values)


lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\small",
    r"\caption{Token-budget-matched robustness comparison across three seeds. Both methods use a 14,000-token cap; actual token consumption may differ.}",
    r"\label{tab:a7-token-matched}",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Method & Accuracy ($\pm$ std.) & Calls/question & Tokens/question \\",
    r"\midrule",
]

for method, names in groups.items():
    summaries = []

    for name in names:
        path = root / name / "summary.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing summary: {path}")

        summaries.append(
            json.loads(path.read_text(encoding="utf-8"))
        )

    if any(s.get("n") != 130 for s in summaries):
        raise ValueError(f"Unexpected question count for {method}")

    acc, acc_std = mean_std(summaries, "accuracy")
    calls, _ = mean_std(summaries, "mean_calls")
    tokens, _ = mean_std(summaries, "mean_tokens")

    lines.append(
        f"{method} & {acc:.3f} $\\pm$ {acc_std:.3f} "
        f"& {calls:.2f} & {tokens:.0f} " + r"\\"
    )

lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    "",
])

out = Path("paper/tables/a7_token_matched.tex")
out.write_text("\n".join(lines), encoding="utf-8")

print(f"Created {out}")
print(out.read_text(encoding="utf-8"))