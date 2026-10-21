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

from common.interfaces import Instance
from common.llm import Session


def solve(instance: Instance, llm: Session, n: int = 5, temperature: float = 0.7) -> str:
    raise NotImplementedError("Faheema: best-of-N majority vote")
