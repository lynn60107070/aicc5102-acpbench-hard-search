"""Structured prompting with majority vote (owner: Lynn). Post-hoc follow-up, added 9 Nov 2026.

Combines the two strongest baselines: sample the structured prompt n times at temperature 0.7 and
return the most common final action, as best-of-N does with the official prompt. n=14 matches the
call budget of search and best-of-N. Report it as a post-hoc follow-up, not as the frozen extension.

Go/no-go, fixed before the validation run: continue to the test split only if, on the validation
split (40 questions, seeds 1-3), the mean accuracy beats both best-of-N (n=14) and greedy
structured prompting by at least 3 questions.
"""

from collections import Counter

from common.interfaces import BudgetExceeded, Instance
from common.llm import Session
from common.validator import parse_action
from methods import structured


def solve(instance: Instance, llm: Session, n: int = 14, temperature: float = 0.7) -> str:
    actions = []
    try:
        for _ in range(n):
            action = parse_action(structured.solve(instance, llm, temperature=temperature))
            if action:
                actions.append(action)
    except BudgetExceeded:
        pass  # vote over the samples collected so far
    if not actions:
        return ""
    return Counter(actions).most_common(1)[0][0]
