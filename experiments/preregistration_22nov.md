# Pre-registration: 70B follow-up controls and re-created validation runs (22 Nov 2026)

Written and committed **before** any of the runs below were started. Every run listed here is
reported, whatever it shows. No prompt, parameter or method is changed after seeing a result.

## A. 70B follow-up controls (test split, 130 questions)

Why: the controls and the post-hoc follow-up (Appendix D) were run on Llama 3.1 8B only. These
runs repeat them unchanged on Llama 3.1 70B, the paper's second comparison model. This is a
replication at a second model size, not tuning: no method, prompt or setting differs from 8B.

Setup: `--model meta-llama/llama-3.1-70b-instruct`, `ACP_PROVIDER="Amazon Bedrock"` (as for every
70B run), same prompts, same validator, same seeding (`SEED_STRIDE`).

| Run name | Command (seed s = 1, 2, 3 where shown) |
|---|---|
| `structured_70b` | `--method structured` (greedy, one run) |
| `best_of_n14_70b_s{s}` | `--method best_of_n --param n=14 --seed s` |
| `followup_sv_70b_s{s}` | `--method structured_vote --max-calls 14 --seed s` |

Analysis plan (fixed now):
1. Primary: structured + vote vs best-of-N at 70B, same 14 calls. Per-seed exact McNemar and the
   seed-pooled sign-flip test of `experiments/followups.py`, Holm-corrected together with the
   existing ten pooled tests.
2. Secondary: structured + vote vs 70B direct generation (0.377, our reproduction) and vs the
   paper's 70B value 0.42 (one-sample binomial per seed; the paper's value is itself an estimate).
3. Controls: best-of-N and greedy structured vs 70B direct (per-seed McNemar).
4. Report accuracy (mean ± sample std over seeds), calls and tokens per question, as for 8B.

No go/no-go gate on validation: the method is unchanged from 8B, where it passed its gate.
Whatever the 70B result, the 8B conclusions stand as reported; a negative 70B result is reported
as "the gain does not transfer to 70B".

## B. Re-created validation comparison behind the frozen design (validation split, 40 questions)

Why: the design was frozen on 23 Oct after comparing evaluators on validation, but those runs
were not kept. These runs re-create the comparison with the final, frozen code (later commits
changed only bug fixes and logging), so they document the choice; they cannot reproduce the
original runs exactly.

Model 8B (`ACP_PROVIDER=CoreWeave`), `--split validation`, `--param k=3 --max-calls 14`, seeds 1–3:
`scoring=llm` (frozen, d=2), `steps` (d=2), `vote` (d=1, as in the ablations), `fixed` (d=2),
`random` (d=2). Run names `val_search_{scoring}_s{s}`.

Reporting: correct answers out of 40 per seed and the mean, for each evaluator. If the frozen
0–10 evaluator is not the best here, the paper says so.
