"""How accurate is vote_check's applicability check? Analysis only: uses the validator after the runs.

    python scripts/followup_check_quality.py
"""
import json, sys
from collections import Counter
sys.path.insert(0, ".")
from common import validator
from common.data import load_docs
import re  # same parser as methods/vote_check.py
docs = {str(d["id"]): d for d in load_docs(split="validation")}
def verdict(text):
    m = re.findall(r"applicable\W{0,3}\s*(yes|no)", text, flags=re.I)
    if m: return m[-1].lower() == "yes"
    v = re.findall(r"(cannot be applied|can not be applied|not applicable|can be applied|is applicable)", text, flags=re.I)
    return bool(v) and v[-1].lower() in ("can be applied", "is applicable")
tp = fp = tn = fn = 0
for s in (1, 2, 3):
    for l in open(f"results/final/followup_vote_check/val_vc_s{s}/rows.jsonl", encoding="utf-8"):
        r = json.loads(l); doc = docs[str(r["id"])]
        acts = [a for a in (validator.parse_action(x["text"]) for x in r["replies"] if x["tag"] == "best_of_n") if a]
        ranked = [a for a, _ in Counter(acts).most_common()]
        checks = [x["text"] for x in r["replies"] if x["tag"] == "check"]
        for a, t in zip(ranked, checks):
            yes = verdict(t); truly = validator.score(doc, a).error != "inapplicable"
            if yes and truly: tp += 1
            elif yes: fp += 1
            elif truly: fn += 1
            else: tn += 1
tot = tp + fp + tn + fn
print(f"{tot} checks | truly applicable {(tp+fn)/tot:.2f} | check says yes {(tp+fp)/tot:.2f}")
print(f"impossible moves caught: {tn}/{tn+fp} = {tn/max(tn+fp,1):.2f} | possible moves wrongly rejected: {fn}/{tp+fn} = {fn/max(tp+fn,1):.2f}")
