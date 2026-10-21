# Team guide: start here

Everything shared (data loading, the official prompt, the LLM client, scoring, the runner) already
works and is tested. You only build your own part.

## 1. One-time setup (Windows, cmd)

```bat
git clone <repo url>
cd <repo folder>
py -3.12 -m venv acp-env
acp-env\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and put in the team's API key. Use the same model and endpoint as everyone else.
Then check everything works (offline, about 10 s, no key needed):

```bat
python -m unittest -v
```

All 7 tests must pass. They check that our prompt is byte-identical to lm-eval's and that the
official validator gives the right verdicts on known answers.

Then try one live call (3 instances, about 1 cent):

```bat
python -m experiments.run --method direct --limit 3
```

## 2. Who owns what

You edit only the files you own. To change someone else's file, ask them or open a PR they review.

| Owner | Files | Job |
|---|---|---|
| **Lynn** (lead) | `methods/search.py`, `methods/direct.py`, `scripts/` | Reproduction setup, the search extension |
| **Faheema** | `methods/structured.py`, `methods/best_of_n.py`, `related_work/` | Verify the reproduction, both control baselines, related work |
| **Aisha** | `experiments/`, `common/` (with Lynn) | Runner, budget/token accounting, ablation sweeps, statistics, README |
| **Everyone** | `common/interfaces.py` | Change only when all three agree |

## 3. The contract

Every method is one function in `methods/`:

```python
def solve(instance: Instance, llm: Session, **params) -> str:   # text containing one action
```

- `instance` has only `id`, `domain`, `context`, `question`: no PDDL and no answer key.
- Call the model only through `llm.chat(messages, temperature=..., tag="...")`. It counts calls
  and tokens, enforces the budget and caches responses. Always pass a `tag` (e.g. `"propose"`,
  `"vote"`) so cost can be broken down per step.
- **Never** use `common.validator.score` in a method. That is oracle leakage and invalidates
  the result. `parse_action` (the output grammar only) is allowed.
- If the budget runs out, `llm.chat` raises `BudgetExceeded`. Catch it and return your best answer.
- Register new methods in `methods/__init__.py`, then run them with
  `python -m experiments.run --method <name> --param key=value`.

## 4. Running experiments

```bat
python -m experiments.run --method direct                          :: full 130 instances
python -m experiments.run --method best_of_n --param n=5 --seed 1 --limit 20
python -m experiments.run --method search --param k=3 --param d=2 --max-calls 12 --seed 2
```

- Outputs go to `results/runs/<name>/` (`rows.jsonl` per instance + `summary.json`).
  `results/runs/` is git-ignored scratch space. Copy results that go in the paper to
  `results/final/` and commit them.
- Responses are cached in `%USERPROFILE%\.cache\aicc5102_acp\`, so re-running is free and returns
  the same outputs. Different `--seed` values give different samples.
- Sampled methods use 3 seeds (1, 2, 3). Direct generation is deterministic (temperature 0).
- Use `--limit 10` while developing, and the full set only for real results.

## 5. Git workflow

1. Work on your own branch: `lynn/search`, `faheema/baselines`, `aisha/experiments`.
2. Pull `main` into your branch often: `git pull origin main`.
3. Run `python -m unittest` before every pull request.
4. Open a PR into `main`. Another teammate reviews it before merging. Reviewing also prepares
   you for the defence, where everyone must explain the whole project.
5. Never commit `.env` or API keys.

## 6. Things you need to know

- **Log every deviation from the paper.** If anything in your runs differs from the paper's
  setup (model host, prompt, settings, data version, a failed run), add a row to the
  "Reproduction fidelity log" in `README.md` section 5. That table becomes the paper's
  discrepancy and limitations sections, so nothing should be fixed silently.
- **Plain `lm-eval` crashes on Windows** when scoring this task: the validator reopens an open
  temp file. Our `common/validator.py` and `scripts/lm_eval_win.py` fix it. Use
  `scripts\reproduce_official.bat` for the official harness run.
- **Groq retired Llama 3.1 8B** (16 Aug 2026). We use OpenRouter (see `.env.example`).
- The task has **130 instances**, 10 per domain across 13 domains. The paper reports **0.25** for
  Llama 3.1 8B. A match means our score falls in about **[0.176, 0.324]**.
- Scoring labels every wrong answer as `format` (no parsable action), `inapplicable`, or
  `not_closer` (applicable, but not on an optimal plan).

## 7. Your first tasks

- **Faheema:** run `scripts\reproduce_official.bat` and `python -m experiments.run --method direct`.
  Both should give the same score, and it should land in [0.176, 0.324]. Then start `structured.py`.
- **Aisha:** read `experiments/run.py` and `common/`. Add the paired comparison script
  (McNemar on two `rows.jsonl` files) and a seed-sweep / ablation driver.
- **Lynn:** check by hand whether the 8B model can predict next states on a few instances
  (this decides whether the fallback model is needed), then build `search.py`.
