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

if os.name == "nt":
    tempfile.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete_on_close=False)

from lm_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli_evaluate())
