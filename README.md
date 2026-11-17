# AICC5102 Project: LLM-Guided Search for Open-Ended Planning on ACPBench-Hard

**Team (Section 1, Dr. Elyor Kodirov):** Lynn Younes (60107070, Lead), Faheema Hassan Kudukky (60327609), Aisha Suleiman Shaik (60328939)
**Paper approved:** 29 Sep 2026

This file is the single index for the project: links, setup, how the smoke test was run, and deadlines.

---

## 1. Folder map

| Path | What it is |
|---|---|
| `proposal/AICC_project_proposal.pdf` | Submitted proposal (`AICC_project_proposal (1).pdf` is a byte-identical duplicate) |
| `proposal/smoke_cmd.png`, `smoke_score.png` | Screenshots used as "proof the code runs" |
| `requirements/AICC5102_Research_Reproduction_and_Extension_v1 - Copy.pdf` | Assignment brief (what to submit, deadlines, marks) |
| `requirements/Course-Section-Guide_Section-1_v3.pdf` | Section guide (schedule, paper list, AI-use rules) |
| `requirements/CCIT AICC5102 Fall 2026.pdf` | Course master syllabus |
| `results/groq_smoke1/`, `results/groq_smoke3/` | **Copies** of the smoke-test outputs (originals are in `C:\Users\lynny\results\`) |
| `env/acp-env-requirements.txt` | `pip freeze` of the venv the smoke test ran in |
| **`CONTRIBUTING.md`** | **Team guide: setup, who owns what, the method contract, git workflow** |
| `common/` | Shared, tested infrastructure: method contract, LLM client (cache + budget), data + official prompt, validator (Windows-fixed) |
| `methods/` | `direct.py` (working baseline), `structured.py` + `best_of_n.py` (Faheema), `search.py` (Lynn) |
| `experiments/run.py` | Runs any method on the benchmark and scores it (Aisha) |
| `scripts/` | Official lm-eval reproduction: `reproduce_official.bat` (Windows), `reproduce_official.sh` (macOS/Linux), both via `lm_eval_win.py` |
| `tests/` | Offline tests: `python -m unittest -v` |

---

## 2. Core links (download these)

### The paper we reproduce
| Item | Link |
|---|---|
| **ACPBench Hard** (Kokel, Katz, Srinivas, Sohrabi): the paper we reproduce | https://arxiv.org/abs/2503.24378 (PDF: https://arxiv.org/pdf/2503.24378) |
| ACPBench (original, AAAI 2025), background only | https://arxiv.org/abs/2410.05669 (PDF: https://arxiv.org/pdf/2410.05669) |
| Project homepage | https://ibm.github.io/ACPBench/ |

> The proposal cites ACPBench-Hard as "ICLR 2026". The harness README cites it as the arXiv preprint. Check the venue before writing the final bibliography.

### Code
| Item | Link |
|---|---|
| Official ACPBench repo (cited in proposal) | https://github.com/IBM/ACPBench |
| **LM Evaluation Harness**: the code we actually ran | https://github.com/EleutherAI/lm-evaluation-harness |
| ACPBench task configs inside the harness (prompts, 2-shot examples, validator) | https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/acpbench |
| lm-eval on PyPI (version we used: 0.4.13) | https://pypi.org/project/lm-eval/0.4.13/ |

### Data (Hugging Face)
| Item | Link |
|---|---|
| **`ibm-research/acp_bench`**: what the harness actually loads (`dataset_path` in the task YAML), config `acp_nexta_gen`, split `test` | https://huggingface.co/datasets/ibm-research/acp_bench |
| `ibm/ACPBench`: the dataset linked in the proposal | https://huggingface.co/datasets/ibm/ACPBench |

> ⚠️ **Mismatch:** the proposal links `ibm/ACPBench`, but the run used `ibm-research/acp_bench`. In the final paper, cite the one actually used (`ibm-research/acp_bench`) or explain both.

### Model for the reproduction
| Item | Link |
|---|---|
| Llama 3.1 8B Instruct (gated: request access on HF first) | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| Same model via Ollama (`ollama pull llama3.1:8b`) | https://ollama.com/library/llama3.1 |
| **OpenRouter (chosen default host, ~$0.05/1M input tokens)** | https://openrouter.ai/meta-llama/llama-3.1-8b-instruct |
| Hugging Face Inference Providers (alternative: deepinfra, novita, nscale, featherless) | https://router.huggingface.co/v1 |
| Groq deprecations: **Llama 3.1 8B retired 16 Aug 2026**, so Groq can't serve the reproduction | https://console.groq.com/docs/deprecations |

> No GPU on Lynn's laptop, so running locally with Ollama is not practical. OpenRouter may route requests to different providers, which can change quantization. Note this as a possible source of discrepancy (the proposal already lists quantization).

### Related work for the extension (named in the proposal)
| Paper | Paper link | Code |
|---|---|---|
| Tree of Thoughts (Yao et al., NeurIPS 2023) | https://arxiv.org/abs/2305.10601 | https://github.com/princeton-nlp/tree-of-thought-llm |
| Reasoning via Planning, RAP (Hao et al., EMNLP 2023) | https://arxiv.org/abs/2305.14992 | https://github.com/maitrix-org/llm-reasoners |

### Related work from the course paper list (worth citing)
| Paper | Link |
|---|---|
| [2] Thought of Search (Katz, Kokel, Srinivas, Sohrabi, NeurIPS 2024): same IBM authors, LLMs + search | https://proceedings.neurips.cc/paper_files/paper/2024/file/fa080fe0f218871faec1d8ba20e491d5-Paper-Conference.pdf |
| [4] Classical Planning with LLM-Generated Heuristics (Corrêa, Pereira, Seipp, NeurIPS 2025): Lynn's presentation | https://arxiv.org/pdf/2503.18809 |
| [1] ReAct (Yao et al., ICLR 2023) | https://arxiv.org/pdf/2210.03629 |
| [12] LLM+P (Liu et al., 2023) | https://arxiv.org/pdf/2304.11477 |
| [13] Generalized Planning in PDDL with LLMs (Silver et al., AAAI 2024) | https://arxiv.org/pdf/2305.11014 |
| [14] PlanBench (Valmeekam et al., NeurIPS 2023) | https://arxiv.org/pdf/2206.10498 |
| [17] AI Agents That Matter (Kapoor et al., 2024): cost-matched evaluation, supports our budget matching | https://arxiv.org/pdf/2407.01502 |

### Paper template
| Item | Link |
|---|---|
| NeurIPS 2026 LaTeX template (use the **preprint** option; 6 pages of main content) | https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip |

---

## 3. How the smoke test was reproduced (recovered from the machine)

**Where:** from `C:\Users\lynny` in **cmd.exe**: the prompt in the screenshot is `(acp-env) C:\Users\lynny>` and the command uses `^` line continuation. cmd keeps no history, so the install steps below are reconstructed from the venv contents.

**Environment:** `C:\Users\lynny\acp-env`, created with Python 3.12.2 (`python -m venv C:\Users\lynny\acp-env`), on 4 Oct 2026.

Key packages (full list in `env/acp-env-requirements.txt`):
`lm_eval==0.4.13`, `datasets==5.0.1`, `transformers==4.57.6`, `torch==2.14.1+cpu`, `tarski==0.8.2`, `pddl==0.4.2`, `lark==1.1.9`, `kstar-planner==1.6.0`.

> Note: lm-eval's `acpbench` extra pins `kstar-planner==1.4.2`, but 1.6.0 is installed, so the planner was probably installed separately. If validator results look off, try 1.4.2.

**Recreate the environment (cmd.exe):**
```bat
python -m venv acp-env
acp-env\Scripts\activate
pip install -r requirements.txt
REM or the exact frozen environment:
pip install -r env\acp-env-requirements.txt
```

**Groq API key:** the `local-chat-completions` backend reads `OPENAI_API_KEY`:
```bat
set OPENAI_API_KEY=<your Groq key>
```

**Smoke command (3 instances):**
```bat
lm-eval --model local-chat-completions ^
  --model_args model=openai/gpt-oss-120b,base_url=https://api.groq.com/openai/v1/chat/completions ^
  --tasks acp_nexta_gen --apply_chat_template --limit 3 --log_samples --output_path results\groq_smoke3
```

**What came out (`results/groq_smoke3`, run 4 Oct 2026 12:32):**
- Score 0.3333 ± 0.3333 (1/3), filter `acp_grammar_parse`, 2-shot, temperature 0, ~138 s.
- Per instance: doc 0 → parsed `None` (score 0), doc 1 → `(sail l3 l0)` (score 1), doc 2 → parsed `None` (score 0).
- The 2 failures were **empty responses** (`""`) from gpt-oss-120b, not badly formatted answers. The reasoning model likely spent its output on hidden reasoning. This won't apply to Llama 3.1 8B.
- The smoke test only worked because none of the 3 answers needed the planner check. **Plain `lm-eval` crashes on Windows** for any answer in the "maybe" list (83 of the 130 questions have one): `acp_utils.py` writes to a `NamedTemporaryFile` while it is still open. Fixed in `common/validator.py` and `scripts/lm_eval_win.py` (`delete_on_close=False`), verified against the answer key.
- `groq_smoke1` is an earlier run from 12:01, the same day.

---

## 4. Task facts needed for the reproduction

- **Task:** `acp_nexta_gen` (tag `acp_bench_hard`), config in `lm_eval/tasks/acpbench/gen_2shot/next_act.yaml`.
- **Test set size:** 130 instances (`n-samples.original` in the results JSON).
- **Prompt:** `**Question**: {context} What is the next action that takes us towards the goal? Each action starts with an opening parenthesis and ends with closing parenthesis. Provide only the action. **Final Answer**:`
- **Generation:** greedy (`temperature 0`), `max_gen_toks 1000`, stops on `\n\n`, `**Question**:`, `Q:`.
- **Validator:** `acp_utils.py` (tarski + K* planner). Correct when the action is applicable and h(s) − h(t) = 1.
- **Target:** the paper reports **0.25** for Llama 3.1 8B Instruct. With n = 130, the 95% binomial CI around 0.25 is about **[0.176, 0.324]** (0.25 ± 1.96·√(0.25·0.75/130)). A reproduced score inside this range counts as a match.
- **Full run, two independent ways (both should agree):**
  ```bat
  scripts\reproduce_official.bat                     :: official lm-eval harness (macOS: bash scripts/reproduce_official.sh)
  python -m experiments.run --method direct          :: our pipeline, same prompt byte-for-byte
  ```

---

## 4b. Reproduce every result in the paper

Setup: on Windows, clone with `git -c core.longpaths=true clone <repo-url>` (or into a short
path such as `C:\acp`); otherwise some files under `results/` have paths too long to check out.
Create the environment (section 3), copy `.env.example` to `.env` and add an OpenRouter key.
`.env` already sets Llama 3.1 8B with `ACP_PROVIDER=CoreWeave` (bf16). For 70B, also set
`ACP_PROVIDER=Amazon Bedrock` and pass `--model meta-llama/llama-3.1-70b-instruct`. Commands are for
cmd.exe; on macOS use `/` in paths. Every run writes `rows.jsonl` + `summary.json` to
`results/runs/<name>/`; the runs used in the paper are copied in `results/final/`.

```bat
:: Reproduction (Table 1): official harness, then our pipeline with the same prompt
scripts\reproduce_official.bat
python -m experiments.run --method direct --name repro_llama31_8b_direct_bf16_coreweave
python -m experiments.run --method direct --model meta-llama/llama-3.1-70b-instruct --name repro_llama31_70b_direct_bedrock

:: Main comparison (Table 2), 8B; search and best-of-N with seeds 1, 2, 3
python -m experiments.run --method structured --name structured_8b
python -m experiments.run --method best_of_n --param n=14 --seed 1 --name best_of_n14_8b_s1
python -m experiments.run --method search --param k=3 --param d=2 --max-calls 14 --seed 1 --name main_search_8b_s1
:: 70B search: same as the line above with --model meta-llama/llama-3.1-70b-instruct --name main_search_70b_s1

:: Token-matched rows and ablations (each line x seeds 1-3)
python -m experiments.sweep experiments\a7_token_matched.txt --seeds 1 2 3
python -m experiments.sweep experiments\ablations.txt --seeds 1 2 3

:: Post-hoc follow-ups (Appendix D): validation first, test only for the one that passed
python -m experiments.run --method vote_check --split validation --max-calls 14 --seed 1 --name val_vc_s1
python -m experiments.run --method structured_vote --split validation --max-calls 14 --seed 1 --name val_sv_s1
python -m experiments.run --method structured_vote --max-calls 14 --seed 1 --name followup_sv_8b_s1
python -m experiments.run --method structured_vote --max-calls 14 --max-tokens 14000 --seed 1 --name followup_sv14k_8b_s1
:: Appendix D analysis numbers (validator used after the runs only; no API calls)
python scripts\followup_headroom.py
python scripts\followup_check_quality.py

:: Paper tables (paper/tables/) from the saved runs; no API calls
python -m experiments.analyze --runs results\final\repro_* results\final\structured_8b results\final\best_of_n14_8b_s* results\final\main_search_* results\final\a7\* results\final\ablations\abl8b_*_s* --out results\analysis
python -m unittest
```

The last two commands need no API key: they rebuild every table in the paper from `results/final/`.
`experiments.sweep` skips runs that already have a `summary.json` under `results/final/`, so to
re-run them from scratch, rename `results/final/` first. Full re-runs cost API credit (the whole
project, including development runs, spent about $5–6 on OpenRouter).

---

## 5. Reproduction fidelity log (feeds the paper's discrepancy + limitations sections)

**Principle:** match the paper as closely as possible, record every deviation as it happens, and report honestly what could not be reproduced and why. A match means the score falls within the 95% interval, not that it equals the paper's number exactly.

**Scope**
| Target | Paper value | Status |
|---|---|---|
| Llama 3.1 8B, Next Action (required) | 0.25 (Table 1) | ✅ **0.223** (29/130, 95% CI 0.15–0.29), bf16 via CoreWeave, 21 Oct 2026 → match. `results/final/repro_llama31_8b_direct_bf16_coreweave/` |
| Llama 3.1 8B, all 8 tasks (stretch: whole Table 1 row) | Table 1 row | optional, `--tasks acp_gen_2shot` via `scripts/lm_eval_win.py` |
| Llama 3.1 8B, Progression (state prediction; informs the search design) | 0.40 (Table 1) | ✅ **0.400** (official harness, bf16 CoreWeave, 21 Oct) → exact match. `results/final/repro_prog_llama31_8b/` |
| Llama 3.1 70B, Progression | 0.65 | ✅ **0.608** (official harness, Amazon Bedrock, 21 Oct) → match (95% CI ±0.08). `results/final/repro_prog_llama31_70b/` |
| Llama 3.1 70B, Next Action (second comparison model for search) | 0.42 | ✅ **0.377** (49/130, Amazon Bedrock, 22 Oct) → match (interval for 0.42: 0.335–0.505). `results/final/repro_llama31_70b_direct_bedrock/` |
| Closed models (GPT-4o, GPT-4o mini, o1-mini, o1-preview) | — | not reproducible (see below) |
| Llama 3.1 8B, Next Action (Reproduction [F2]) | 0.25 (Table 1) | ✅ **0.2231** (official harness run, 22 Oct 2026) → match. `results/official_reproduction/` |

**Known deviations from the paper's setup** (add a row whenever anything else differs)
| # | Paper | Ours | Why | Expected effect |
|---|---|---|---|---|
| D1 | Model access not specified per model (API, or HF transformers on 2×A100) | Llama 3.1 8B via OpenRouter, pinned to CoreWeave (bf16) with `ACP_PROVIDER` | No local GPU; Groq retired the model 16 Aug 2026. Unpinned, OpenRouter mixed providers: run 1 (21 Oct) went to Groq 97 / DeepInfra 23 / Novita 9 / Cloudflare 1, and 3 of those 4 serve fp8 | Mixed run 0.192 vs bf16 run 0.223 (paper 0.25). The two runs disagree on 16 questions (6 right only in run 1, 10 right only in run 2): precision visibly changes answers, but the accuracy difference is not significant (exact McNemar p = 0.45, `experiments/compare.py`) |
| D1b | (same as D1) | Llama 3.1 70B via OpenRouter → Amazon Bedrock | Only 70B providers are DeepInfra (fp8, blocked) and Bedrock (precision unspecified) | Progression 0.608 vs paper 0.65, inside the interval |
| D2 | 5 stop sequences (A.5) | lm-eval's API mode sends only the first 4 (drops `"Q:"`) | OpenAI-style APIs accept at most 4 | Probably negligible |
| D3 | Chat template use not stated | `--apply_chat_template` (chat messages) | Required by the chat API | All four results inside the interval, so kept |
| D4 | Paper's own YAML is in its supplementary material | lm-eval's built-in `acpbench` task config | Supplementary not yet checked | Likely identical (same authors) — verify |
| D5 | Paper ran on Linux | Windows, with the temp-file fix to the official validator | Plain lm-eval crashes on Windows | None: same scoring code, verified against the answer key |
| D6 | Dataset as of paper (v2, Feb 2026) | `ibm-research/acp_bench` as of download date | Dataset may have been updated | Revision `05e5883`, downloaded 4 Oct 2026. Size matches (130); the independent macOS run gave identical answers |

**What can't be reproduced, and why (for the limitations section)**
- Closed-model rows: API versions change or are retired (o1-preview is gone), so the original models no longer exist.
- Exact numbers: greedy decoding isn't bit-for-bit deterministic across hardware, batching and precision. With n = 130, one flipped answer moves the score by 0.008.
- Unreported settings (D1, D3) can't be copied exactly.
- Very large models (Llama 3.1 405B) and reasoning models: cost/hardware beyond the project's budget.
- Dataset generation: we re-evaluate the released dataset; we don't regenerate the questions.

---

## 5b. Main extension results (test, 130 questions, frozen search, seeds 1–3)

Quick reference. The paper's tables come from `experiments/analyze.py`.

| | Llama 3.1 8B | Llama 3.1 70B |
|---|---|---|
| Direct generation | 0.223 | 0.377 |
| Structured prompting (1 call, greedy) | 0.262 | — |
| Best-of-N, N = 14 (same calls as search) | 0.236 ± 0.012 | — |
| **Search** (`scoring=llm k=3 d=2 prune`, 14 calls) | **0.159 ± 0.012** | **0.362 ± 0.015** |
| McNemar p vs direct, per seed | 0.17 / 0.27 / 0.09 | 0.57 / 0.85 / 1.00 |
| Correct action among root candidates (analysis only) | 0.313 | 0.590 |
| First proposal alone (analysis only) | 0.179 | 0.280 |
| Score separation AUC, root candidates (0.5 = chance) | 0.578 | 0.527 |
| Calls per question | 13.4 | 11.4 |

Runs: `results/final/main_search_{8b,70b}_s{1,2,3}/`. Finding: search never beats direct; the model often
proposes a correct action but its own scores barely tell it apart from wrong ones, even at 70B.

Post-hoc follow-up (paper Appendix D, after the progress check): structured prompting with a majority
vote over 14 samples reaches **0.331 ± 0.013** on the 8B test set (paper's direct result: 0.25), at the same
14 calls. It is significant against direct (p ≤ 0.05 in every seed) but not against greedy structured
prompting, and uses 26% more tokens than best-of-N. Runs: `results/final/followup_structured_vote/`.

## 6. Deadlines (Section 1)

Team tasks (F1–F7, A1–A9, L1–L8) and checkpoints CP1–CP6 (whole submission done by Sun 1 Nov) are in [SCHEDULE.md](SCHEDULE.md).

| Stage | Date | Weight (of course) |
|---|---|---|
| Proposal | 29 Sep 2026 ✅ | 4% |
| **Progress check (required, ungraded): show code + early results** | **Tue 3 Nov 2026** | n/a |
| Paper (PDF + LaTeX) and code ZIP with README + results | Tue 17 Nov 2026 | Paper 6% + Code/tests 8% (section guide says report 10%) |
| Presentation + individual defence | **Tue 24 Nov 2026** | 8% |

**Final submission must include:** six-page NeurIPS paper (abstract, research question, related work, method and extension, experiments, limitations, conclusion; cite the paper **and** the code); LaTeX source; code ZIP with a README of reproduce commands; names, IDs, section, each member's contribution; an **AI use statement** (tool, what you asked, what it gave back, what you changed).

**Fairness rules from the brief:** same problems for baseline and extension; ablations; **≥ 3 seeds** for random methods, reporting mean ± std; report negative results honestly.

---

## 7. Early to-dos (all done)

- [x] Choose how to serve Llama 3.1 8B → OpenRouter (Groq retired the model; no local GPU).
- [x] Shared pipeline + Windows validator fix + tests (see `CONTRIBUTING.md`).
- [x] OpenRouter key in `.env`; live test passed.
- [x] Full 130-instance baseline: 0.223 bf16 (match), 0.192 mixed-provider run kept as evidence for D1.
- [x] Faheema: independent check of the reproduction through the official harness (`scripts\reproduce_official.bat`, or `bash scripts/reproduce_official.sh` on macOS).
- [x] Push the repo to GitHub and add Faheema and Aisha; fill their usernames into `.github/CODEOWNERS`.
- [x] Settle the HF dataset citation: `ibm-research/acp_bench`, revision `05e5883` (D6).
- [x] Prototype the best-first search extension (k candidates, depth d, call budget); frozen 23 Oct.
