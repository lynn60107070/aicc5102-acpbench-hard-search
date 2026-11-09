"""Best-of-N with an LLM applicability check (owner: Lynn). Post-hoc follow-up to the search.

Added 9 Nov 2026, after the main results: report it as a post-hoc follow-up, not as the frozen
extension. Motivation (offline analysis of the existing runs): most wrong best-of-N answers are
inapplicable actions, and the search's predictor almost never says NOT APPLICABLE. This asks the
narrower question directly, as a precondition checklist.

    sample  = n answers with the official prompt at temperature 0.7, as in best-of-N    (n calls)
    rank    = distinct parsed actions by vote count (ties: first sampled wins)
    check   = for the top m actions in rank order, the LLM lists the action's          (≤ m calls)
              requirements and checks each against the state; stop at the first it accepts
    answer  = the first accepted action, else the top-voted action

n=11, m=3 uses at most 14 calls, the same budget as search and best-of-N. With the same seed the
n samples are the first n samples of best-of-N (shared cache), so the comparison is paired.
Only the LLM judges applicability: no planner, simulator or validator.

Go/no-go, fixed before the validation run: continue to the test split only if, on the validation
split (40 questions, seeds 1-3), the mean accuracy beats best-of-N (n=14) by at least 3 questions.
"""

import re
from collections import Counter

from common.data import OFFICIAL_MAX_TOKENS, OFFICIAL_STOP, official_messages
from common.interfaces import BudgetExceeded, Instance
from common.llm import Session
from common.validator import parse_action


def _applicable(llm: Session, instance: Instance, action: str) -> bool:
    prompt = (f"{instance.context}\n\n"
              f"Can the action {action} be applied in the current state? First list every condition "
              f"that must hold for this action, based on the domain description. Then check each "
              f"condition against the current state, one per line, ending each line with 'holds' or "
              f"'fails'. Then end with a final line of the form: Applicable: yes or Applicable: no")
    text = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=500,
                    tag="check")
    matches = re.findall(r"applicable\W{0,3}\s*(yes|no)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower() == "yes"
    # No final line: use the last verdict phrase in the reasoning instead
    verdicts = re.findall(r"(cannot be applied|can not be applied|not applicable|can be applied|is applicable)",
                          text, flags=re.IGNORECASE)
    return bool(verdicts) and verdicts[-1].lower() in ("can be applied", "is applicable")


def solve(instance: Instance, llm: Session, n: int = 11, m: int = 3,
          temperature: float = 0.7) -> str:
    messages = official_messages(instance)
    actions = []
    try:
        for _ in range(n):
            action = parse_action(llm.chat(messages, temperature=temperature,
                                           max_tokens=OFFICIAL_MAX_TOKENS, stop=OFFICIAL_STOP,
                                           tag="best_of_n"))
            if action:
                actions.append(action)
    except BudgetExceeded:
        pass
    if not actions:
        return ""
    ranked = [a for a, _ in Counter(actions).most_common()]  # Counter keeps first-seen order on ties
    try:
        for action in ranked[:m]:
            if _applicable(llm, instance, action):
                return action
    except BudgetExceeded:
        pass
    return ranked[0]
