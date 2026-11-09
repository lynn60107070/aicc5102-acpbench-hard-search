"""Offline headroom analysis of the existing test runs (no API calls). Uses the validator for analysis only.

    python scripts/followup_headroom.py
"""
import json, sys
from collections import Counter
sys.path.insert(0, ".")
from common import validator
from experiments.analyze import load_documents

docs = load_documents()
rows = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

print("== best-of-N 14 (8B) ==")
for s in (1, 2, 3):
    R = rows(f"results/final/best_of_n14_8b_s{s}/rows.jsonl")
    acc = cov = vote_applicable = 0
    errs = Counter()
    for r in R:
        doc = docs[str(r["id"])]
        acts = [a for a in (validator.parse_action(x["text"]) for x in r["replies"]) if a]
        verdicts = {a: validator.score(doc, a) for a in set(acts)}
        acc += r["correct"]
        errs[r["error"]] += 1
        cov += any(v.correct for v in verdicts.values())
        # upper bound for a perfect applicability filter: vote only over truly applicable actions
        appl = [a for a in acts if verdicts[a].error != "inapplicable"]
        if appl and verdicts[Counter(appl).most_common(1)[0][0]].correct:
            vote_applicable += 1
    n = len(R)
    print(f"s{s}: vote acc {acc/n:.3f} | any of 14 correct {cov/n:.3f} | "
          f"vote over truly-applicable only {vote_applicable/n:.3f} | errors {dict(errs)}")

print("\n== search root: how good is the LLM's own NOT APPLICABLE judgement? (8B) ==")
for s in (1, 2, 3):
    R = rows(f"results/final/main_search_8b_s{s}/rows.jsonl")
    tp = fp = tn = fn = 0  # positive = LLM says applicable
    for r in R:
        doc = docs[str(r["id"])]
        reps = r["replies"]
        i = next((j for j, x in enumerate(reps) if x["tag"] == "propose"), None)
        if i is None:
            continue
        cands = validator.parse_actions(reps[i]["text"])
        preds = []
        for x in reps[i + 1:]:
            if x["tag"] == "propose":
                break
            if x["tag"] == "predict":
                preds.append("NOT APPLICABLE" not in x["text"].upper())
        for a, says_appl in zip(cands, preds):
            truly = validator.score(doc, a).error != "inapplicable"
            if says_appl and truly: tp += 1
            elif says_appl: fp += 1
            elif truly: fn += 1
            else: tn += 1
    tot = tp + fp + tn + fn
    print(f"s{s}: {tot} candidates, truly applicable {(tp+fn)/tot:.2f}; LLM says applicable {(tp+fp)/tot:.2f}; "
          f"of those, truly applicable {tp/max(tp+fp,1):.2f}; of LLM 'NOT APPLICABLE', truly inapplicable {tn/max(tn+fn,1):.2f}; "
          f"inapplicable candidates caught {tn}/{tn+fp} = {tn/max(tn+fp,1):.2f}")
