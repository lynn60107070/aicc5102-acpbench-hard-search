"""Best-of-N with majority vote (owner: Faheema).

Sample N answers at temperature 0.7 and return the most common action. This tests
whether gains come from extra candidates alone.

Rules:
- The vote must NOT use the validator (oracle leakage). Normalising answers with
  common.validator.parse_action is allowed: it is only the output grammar and sees
  no PDDL. Import it as `from common.validator import parse_action`.
- Identical sampled requests automatically get different responses (see common/llm.py).
- Match budgets: N should equal the search method's call budget in the main comparison.
"""

from collections import Counter
from common.data import OFFICIAL_MAX_TOKENS, OFFICIAL_STOP, official_messages
from common.interfaces import BudgetExceeded, Instance  # CHANGED: added BudgetExceeded
from common.llm import Session
from common.validator import parse_action


def solve(instance: Instance, llm: Session, n: int = 5, temperature: float = 0.7) -> str:
    messages = official_messages(instance)
    parsed_actions = []

    try:  # NEW
        for _ in range(n):
            response = llm.chat(
                messages,
                temperature=temperature,
                max_tokens=OFFICIAL_MAX_TOKENS,
                stop=OFFICIAL_STOP,
                tag="best_of_n"
            )
            action = parse_action(response)
            if action:
                parsed_actions.append(action)
    except BudgetExceeded: # keep the samples collected so far and vote over them
        pass

    if not parsed_actions:
        return ""

    # Return the most common successfully parsed action
    counts = Counter(parsed_actions)
    best_action, _ = counts.most_common(1)[0]
    return best_action