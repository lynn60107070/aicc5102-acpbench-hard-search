
"""Run one method across multiple seeds and parameter configurations.

Example:
    python -m experiments.sweep --method search --seeds 1 2 3 \
        --config "k=3,d=2" --config "k=5,d=2" --limit 10 --workers 4
"""

import argparse
import ast
import subprocess
import sys


def parse_config(config: str) -> list[str]:
    """Convert 'k=3,d=2' into ['k=3', 'd=2'] and validate its values."""
    if not config.strip():
        raise argparse.ArgumentTypeError("configuration cannot be empty")

    params = []
    for item in config.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"invalid parameter {item!r}; expected key=value"
            )

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            raise argparse.ArgumentTypeError(
                f"invalid parameter {item!r}; expected key=value"
            )

        try:
            ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Supports plain strings, e.g. scoring=random.

        params.append(f"{key}={value}")

    return params


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a method across multiple seeds and parameter configurations."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3],
        help="seeds to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--config", action="append", type=parse_config, default=None,
        help='parameter configuration, repeatable; e.g. --config "k=3,d=2"',
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-tokens", type=int)

    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    configs = args.config if args.config is not None else [[]]
    total = len(args.seeds) * len(configs)

    print(f"Method: {args.method}")
    print(f"Configurations: {len(configs)} | Seeds: {len(args.seeds)}")
    print(f"Runs to launch: {total} | Workers per run: {args.workers}")

    run_number = 0
    for config in configs:
        for seed in args.seeds:
            run_number += 1
            config_label = "_".join(
                param.replace("=", "").replace(",", "-")
                for param in config
            ) or "default"

            run_name = (
                f"sweep_{args.method}_{config_label}_seed{seed}"
                f"_{args.split}"
                + (f"_limit{args.limit}" if args.limit is not None else "")
                + (f"_calls{args.max_calls}" if args.max_calls is not None else "")
                + (f"_tokens{args.max_tokens}" if args.max_tokens is not None else "")
            )
            if args.ids is not None:
                run_name += "_ids" + "-".join(args.ids)
            command = [
                sys.executable, "-m", "experiments.run",
                "--method", args.method,
                "--seed", str(seed),
                "--split", args.split,
                "--workers", str(args.workers),
                "--name", run_name,
            ]

            if args.model:
                command.extend(["--model", args.model])
            if args.base_url:
                command.extend(["--base-url", args.base_url])
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            if args.ids is not None:
                command.append("--ids")
                command.extend(args.ids)
            if args.max_calls is not None:
                command.extend(["--max-calls", str(args.max_calls)])
            if args.max_tokens is not None:
                command.extend(["--max-tokens", str(args.max_tokens)])

            for param in config:
                command.extend(["--param", param])

            print(
                f"\n[{run_number}/{total}] seed={seed}, "
                f"config={config or '(default parameters)'}",
                flush=True,
            )
            subprocess.run(command, check=True)

    print(f"\nSweep complete: {total} runs launched.")


if __name__ == "__main__":
    main()
