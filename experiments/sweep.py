"""Run several experiment configurations x seeds in one command (A3).

    python -m experiments.sweep experiments/ablations.txt --seeds 1 2 3 --workers 8

Each non-empty, non-comment line of the config file is `<name> <experiments.run arguments>`.
Every (line, seed) pair runs as results/runs/<name>_s<seed>. Runs that already have a
summary.json are skipped, so an interrupted sweep can simply be restarted.
"""

import argparse
import shlex
from pathlib import Path

from common.llm import REPO_ROOT
from experiments import run


def read_config(path: Path) -> list[tuple[str, list[str]]]:
    configs = []
    for line in path.read_text(encoding="utf8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, *args = shlex.split(line)
            configs.append((name, args))
    return configs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="only list the runs")
    args = ap.parse_args(argv)

    for name, run_args in read_config(args.config):
        for seed in args.seeds:
            out_name = f"{name}_s{seed}"
            if (REPO_ROOT / "results" / "runs" / out_name / "summary.json").exists():
                print(f"skip {out_name} (done)")
                continue
            full = run_args + ["--seed", str(seed), "--name", out_name, "--workers", str(args.workers)]
            print(f"=== {out_name}: {' '.join(full)}", flush=True)
            if not args.dry_run:
                run.main(full)


if __name__ == "__main__":
    main()
