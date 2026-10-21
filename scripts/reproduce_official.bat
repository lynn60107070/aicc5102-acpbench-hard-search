@echo off
REM Full official reproduction: lm-eval harness, acp_nexta_gen, 130 instances, 2-shot, greedy.
REM Reads OPENAI_API_KEY from .env. lm-eval cannot pin an OpenRouter provider, so block the
REM fp8 providers in the OpenRouter account settings first (see CONTRIBUTING.md).
REM Uses scripts\lm_eval_win.py because plain lm-eval crashes on Windows (see README).
REM Extra arguments are passed through, e.g.:  scripts\reproduce_official.bat --limit 10

set MODEL=meta-llama/llama-3.1-8b-instruct
set BASE_URL=https://openrouter.ai/api/v1/chat/completions

python scripts\lm_eval_win.py --model local-chat-completions ^
  --model_args model=%MODEL%,base_url=%BASE_URL% ^
  --tasks acp_nexta_gen --apply_chat_template --log_samples ^
  --output_path results\official_reproduction %*
