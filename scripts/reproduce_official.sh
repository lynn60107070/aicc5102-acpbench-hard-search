#!/usr/bin/env bash
# Full official reproduction (macOS/Linux): lm-eval harness, acp_nexta_gen, 130 instances, 2-shot, greedy.
# Same as scripts\reproduce_official.bat. Reads OPENAI_API_KEY from .env; the OpenRouter account
# only allows the bf16 provider. Run from the repo root with the venv active:
#     bash scripts/reproduce_official.sh              (all 130)
#     bash scripts/reproduce_official.sh --limit 10   (extra arguments are passed through)
set -euo pipefail

MODEL=meta-llama/llama-3.1-8b-instruct
BASE_URL=https://openrouter.ai/api/v1/chat/completions

python scripts/lm_eval_win.py --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL}" \
  --tasks acp_nexta_gen --apply_chat_template --log_samples \
  --output_path results/official_reproduction "$@"
