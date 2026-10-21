"""Run the official lm-eval CLI with the Windows temp-file fix applied.

Plain `lm-eval` crashes on Windows (PermissionError) as soon as the model answers with an
action that needs the planner check. Use this exactly like `lm-eval`:

    python scripts\\lm_eval_win.py --model local-chat-completions --model_args ... --tasks acp_nexta_gen ...

On Linux/macOS/Colab plain `lm-eval` works and this wrapper changes nothing.
"""

import functools
import os
import sys
import tempfile
from pathlib import Path

if os.name == "nt":
    tempfile.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete_on_close=False)
    # lm-eval's results table contains "↑", which the Windows console encoding can't print when
    # output is redirected to a file; results are saved before that, but the run then exits with an error.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Read OPENAI_API_KEY from the repo's .env, like the rest of our code.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import load_dotenv  # noqa: E402

load_dotenv()

from lm_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli_evaluate())
