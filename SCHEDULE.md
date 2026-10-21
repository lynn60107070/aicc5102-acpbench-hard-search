# Schedule and checkpoints

**Goal: 90% done by the course progress check on Tue 3 Nov.** That means every experiment is finished,
including ablations. Nov 4–17 is for writing the paper, one robustness run and cleanup.

**Check-in format (15 min, call or in person):** each person answers three questions:
1. What's done? (Show it: a command that runs, a `summary.json`, a table.)
2. What's blocked?
3. What's next before the next checkpoint?

If you're blocked, message the group **the same day**. Don't wait for the next check-in.

Each checkpoint lists what must be **done** by then, not just started.

---

## CP1: Sun 25 Oct (online), setup + reproduction verified
| Who | Done means |
|---|---|
| Faheema | Repo set up, `python -m unittest` passes, live 3-question test works. `scripts\reproduce_official.bat` run on all 130 questions; score compared with our 0.223, and any difference explained in the fidelity log |
| Aisha | Same setup working. Has read `common/` and `experiments/run.py` and can explain the method contract |
| Lynn | State-prediction test done on ~5 questions, so we know whether search uses 8B or 70B. Search v0 skeleton started |

## CP2: Tue 27 Oct (in person, lab), first versions run end to end
| Who | Done means |
|---|---|
| Faheema | `structured.py` works on `--limit 10`. `best_of_n.py` written and works on `--limit 10` |
| Aisha | McNemar script compares two `rows.jsonl` files (tested on the two reproduction runs). Seed/sweep runner can launch one method × 3 seeds |
| Lynn | `search.py` v0 runs end to end on `--limit 10` with `k`, `d`, `scoring`, `predict_state` switches working |

## CP3: Thu 29 Oct (online), baselines complete, search design frozen
| Who | Done means |
|---|---|
| Faheema | Structured prompting full run (130). Best-of-N full runs × 3 seeds (1, 2, 3), copied to `results/final/` |
| Aisha | Budget matching works (calls and tokens). Analysis script outputs accuracy ± std over seeds, per-domain accuracy and error types. Runner made faster (parallel requests) if runs are too slow |
| Lynn | **Search design frozen.** No changes to the method after today, only bug fixes. Main search config chosen on a small dev subset |

## CP4: Sun 1 Nov (online), main results + ablations
| Who | Done means |
|---|---|
| Lynn | Main search runs: 130 × 3 seeds, in `results/final/` |
| Aisha | Ablations run: no LLM scoring, random/fixed expansion, no state prediction, vary k / d / budget |
| Faheema | Related work notes + `related_work/refs.bib` draft. Second check of everyone's results (spot-check rows) |

## CP5: Mon 2 Nov (online), dry run for the progress check
| Who | Done means |
|---|---|
| All | Results table: direct vs structured vs best-of-N vs search, with McNemar tests. Ablation table. 5-minute walkthrough rehearsed; each person can explain every part |
| All | Paper sections assigned (see below) |

## Course progress check: Tue 3 Nov
Show: reproduction (0.223 vs 0.25), main comparison, ablations, fidelity log.

---

## After the progress check
| Date | Milestone |
|---|---|
| Tue 3 – Thu 5 Nov | Act on instructor feedback. Token-matched robustness check (Aisha) |
| Sun 8 Nov | Each person's paper sections drafted (NeurIPS template, 6 pages) |
| Tue 10 Nov | Full paper draft. Everyone reviews everyone's sections |
| Thu 12 Nov | Final figures/tables. README reproduce commands checked on a fresh clone |
| Sun 15 Nov | Final paper + code ZIP ready (2-day buffer) |
| **Tue 17 Nov** | **Submit: paper PDF, LaTeX source, code ZIP** |
| 24 Nov / 1 Dec | Presentation + individual defence |

**Paper sections:** Lynn writes method & extension and the discussion. Faheema writes related work and the reproduction section.
Aisha writes the experimental setup, results and limitations.

## Budget
Keys have spending caps. Develop with `--limit 10`. The full-size runs (CP3–CP4) cost the most:
Lynn will top up the account before CP3, so tell Lynn **by CP2** roughly how many full runs you need.
