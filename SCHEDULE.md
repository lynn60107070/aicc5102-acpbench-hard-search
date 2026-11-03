# Schedule, tasks and checkpoints

**Goal: the whole submission (all experiments, the full paper draft and the code ZIP) is done by
Sun 1 Nov.** Mon 2 Nov is a buffer + rehearsal, and Tue 3 Nov is the course progress check. After that,
only instructor feedback, slides, defence prep and the peer review remain.

We work through weekends. Effort estimates are hands-on time; full runs also need waiting time,
so start them early and work on something else while they run.

---

## Timeline

| Checkpoint | Date | Where the team should be |
|---|---|---|
| Kick-off | Wed 21 Oct | Repo handed over, everyone sets up tonight |
| **CP1** | **Thu 22 Oct** | Everyone set up; reproduction independently verified; search model (8B/70B) decided |
| **CP2** | **Sat 24 Oct** | Every method and every analysis script runs on 10 questions |
| **CP3** | **Mon 26 Oct** | Baselines fully run; **search design frozen**; related work notes done |
| (lab) | Tue 27 Oct | Quick in-person sync, no new deliverable |
| **CP4** | **Wed 28 Oct** | **All experiments done**; tables and figures generated |
| **CP5** | **Fri 30 Oct** | Full paper draft, every section written |
| **CP6** | **Sun 1 Nov** | Paper reviewed by all; code ZIP checked on a fresh clone. **Submission-ready** |
| Buffer | Mon 2 Nov | Fix leftovers; rehearse the 5-minute walkthrough |
| Course check | Tue 3 Nov | Show the complete work |
| Deadline | Tue 17 Nov | Submit (after applying instructor feedback) |

**Daily update** in the group chat, every day by 10 pm (three lines, even if nothing changed):
```
✅ Done today:   (task IDs)
🚧 Blocked on:
➡️ Tomorrow:     (task IDs)
```
Blocked? Message **immediately**. Don't wait for the evening.

**Checkpoint call (15 min):** each person shows the tasks due (a command that runs, a `summary.json`,
a draft section), and we agree on the focus until the next checkpoint. Done means finished, not started.

---

## Tasks

### Faheema: reproduction check, control baselines, related work
| ID | Task | Effort | Due |
|---|---|---|---|
| F1 | Setup: clone, venv, `.env` key, `python -m unittest` passes, `python -m experiments.run --method direct --limit 3` works | 30 min | Wed 21 (tonight) |
| F2 | Independent reproduction check with the **official lm-eval harness** (not `experiments.run`); verify with `python scripts/check_official_run.py results/official_reproduction`: `scripts\reproduce_official.bat` (Windows) or `bash scripts/reproduce_official.sh` (macOS) on all 130. Compare with our 0.223 (`results/final/`) and record it in README §5 | 1 h (+15 min run) | **CP1** |
| F3 | `methods/structured.py`: state → goal → candidate action → progress → final action. Works on `--limit 10` | 2–3 h | **CP2** |
| F4 | `methods/best_of_n.py`: N samples at temperature 0.7, majority vote with `parse_action` (never the validator). Works on `--limit 10` | 1–2 h | **CP2** |
| F5 | ✅ Done 26 Oct (structured 0.262; best-of-N14 0.236 ± 0.012). Full runs: structured (130) and best-of-N (130 × seeds 1, 2, 3), copied to `results/final/` | 1 h (+ run time) | **CP3** |
| F6 | ✅ Done 2 Nov (refs.bib checked against the published versions, PR #27). Related work notes + `related_work/refs.bib`: ACPBench, ACPBench Hard, ToT, RAP, Thought of Search, LLM-generated heuristics, PlanBench, LLM+P, AI Agents That Matter | 4–5 h | **CP3** |
| F7 | ✅ Done 30 Oct (PR #23; fixed and trimmed by Lynn in PR #24). Paper: related work section + reproduction section (with the discrepancy paragraph from the fidelity log) | 3–4 h | **CP5** |

### Aisha: evaluation, budget accounting, ablations, code package
| ID | Task | Effort | Due |
|---|---|---|---|
| A1 | Setup (as F1) + read `common/` and `experiments/run.py` until you can explain the method contract | 1.5 h | **CP1** |
| A2 | `experiments/compare.py`: McNemar test + accuracy difference between two `rows.jsonl` files. Test it on the two reproduction runs in `results/final/` | 2 h | **CP1** |
| A3 | ✅ Implemented **independently by Aisha** (23 Oct, pushed 25 Oct) **and Lynn** (25 Oct, after the CP2 deadline passed with nothing pushed); Lynn's version was merged first and used for all runs: `experiments/sweep.py` (config × seeds, resumable) and `--workers N` parallel runs in `experiments/run.py`, results identical to sequential runs | 3 h | **CP2** |
| A4 | ✅ Done 26 Oct by Aisha (paper tables + figures, PR #18; polish follow-up requested). First version 25 Oct (search tables, figures, candidate rate, AUC; verified against an independent calculation). Still to add: main comparison table with McNemar p-values, ablation table, reproduction table, output to `paper/tables` and `paper/figures`. `experiments/analyze.py`: accuracy mean ± std over seeds, per-domain table, error types, calls/tokens per method → LaTeX tables + figures. Also, for search runs, **"correct candidate proposed"** (analysis only): parse the first `propose` reply in each row's `replies` with `common.validator.parse_actions` and score each candidate with `validator.score`. This is the upper limit search could reach; the gap to search's accuracy is our main finding. Plus **score separation (AUC)** for the root candidates: in `replies`, after the first `propose`, each `predict` is followed by its `score` unless the predict text says NOT APPLICABLE (then no score); match them in order to the parsed candidates. AUC = probability that a random correct candidate gets a higher score than a random wrong one (ties count ½). 0.5 = no better than chance. Report it with the mean score for correct vs wrong | 4 h | **CP2** |
| A5 | Budget matching: best-of-N's N and search's call budget set to equal calls; report tokens too | 1–2 h | **CP3** |
| A6 | ✅ Done 25 Oct **by Lynn** (run with the A3 sweep; results in `results/final/ablations/`). Ablations of the frozen search (frozen 23 Oct: `scoring=llm k=3 d=2 prune=True --max-calls 14`), 8B, test split, 3 seeds each, changing one setting at a time: `scoring=steps`, `scoring=vote` (with `d=1`), `scoring=fixed`, `scoring=random`, `predict_state=False`, `prune=False`, `k=2` / `k=5`, `d=1` / `d=3`, `--max-calls 7` / `28` | 1 h (+ run time) | **CP4** |
| A7 | ✅ Done 29 Oct by Aisha (PR #22; best-of-N N=8 0.228 vs search 0.151 at a shared 14k-token cap). Token-matched robustness check (main comparison with equal tokens instead of equal calls) | 1 h (+ run time) | **CP4** |
| A8 | ✅ Done 30 Oct by Aisha (PR #25, incl. an error-aggregation fix; assembled by Lynn in PR #26). Paper: experimental setup, results, ablations, limitations | 4–5 h | **CP5** |
| A9 | Code ZIP: README reproduce commands tested on a **fresh clone** in a new folder | 2 h | **CP6** |

### Lynn: lead, search extension
| ID | Task | Effort | Due |
|---|---|---|---|
| L1 | State-prediction test on ~5 questions → decide 8B or 70B for search | 1–2 h | **CP1** |
| L2 | `methods/search.py` v0 with all switches (`k`, `d`, `scoring`, `predict_state`), runs on `--limit 10` | 5–6 h | **CP2** |
| L3 | ✅ Done 23 Oct (frozen: scoring=llm, k=3, d=2, prune, 14 calls). Tune on the **validation split** (`--split validation`, 40 questions, no overlap with test), then **freeze the design**: only bug fixes after this | 3–4 h | **CP3** |
| L4 | ✅ Done 24 Oct (8B + 70B). Main search runs: 130 × seeds 1, 2, 3 in `results/final/` | 1 h (+ run time) | **CP4** |
| L5 | ✅ Done 23 Oct. `paper/` LaTeX skeleton in the NeurIPS template, with one file per section | 1 h | **CP3** |
| L6 | ✅ Drafted (personalisation pending). Paper: abstract, introduction, research question, method & extension, discussion, conclusion | 5–6 h | **CP5** |
| L7 | Contributions statement, AI use statement, final assembly + submission | 2 h | **CP6** |
| L8 | Top up the API account at CP2, based on everyone's planned full runs | 10 min | **CP2** |

### Everyone
| ID | Task | Due |
|---|---|---|
| T1 | Tell Lynn how many full runs you'll need (for the budget top-up) | **CP2** |
| T2 | Read the whole paper draft and comment on every section | **CP6** |
| T3 | Be able to explain every part of the project (individual defence, 8%) | Rehearse Mon 2 Nov |

---

## Checkpoint checklists

**CP1 · Thu 22 Oct:** F1 F2 · A1 A2 · L1
**CP2 · Sat 24 Oct:** F3 F4 · A3 A4 · L2 L8 · T1
**CP3 · Mon 26 Oct:** F5 F6 · A5 · L3 L5
**CP4 · Wed 28 Oct:** A6 A7 · L4 (all experiments done)
**CP5 · Fri 30 Oct:** F7 · A8 · L6 (full paper draft)
**CP6 · Sun 1 Nov:** A9 · L7 · T2 (submission-ready)

If a task slips, say so at the checkpoint and **cut scope rather than move the date**. For example, test
fewer k/d values or drop A7. Everything after CP3 depends on L3 (search frozen) being on time.

## After 3 Nov (light work only)
| When | What |
|---|---|
| 4–6 Nov | Apply instructor feedback, then submit (deadline Tue 17 Nov) |
| by 20 Nov | Presentation slides |
| before 24 Nov / 1 Dec | Individual defence prep |
| TBA | Peer review of another team (4%) |
