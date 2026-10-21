@echo off
REM Full official reproduction: lm-eval harness, acp_nexta_gen, 130 instances, 2-shot, greedy.
REM Needs OPENAI_API_KEY set. Edit MODEL / BASE_URL to the endpoint the team agreed on.
REM Uses scripts\lm_eval_win.py because plain lm-eval crashes on Windows (see README).

set MODEL=meta-llama/llama-3.1-8b-instruct
set BASE_URL=https://openrouter.ai/api/v1/chat/completions

python scripts\lm_eval_win.py --model local-chat-completions ^
  --model_args model=%MODEL%,base_url=%BASE_URL% ^
  --tasks acp_nexta_gen --apply_chat_template --log_samples ^
  --output_path results\official_reproduction
