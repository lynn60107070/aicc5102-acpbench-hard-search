"""Structured prompting control (owner: Faheema).

Ask the model to write out the current state, the goal, the candidate action and the
progress it makes before giving its final action. This tests whether gains come from
extra reasoning alone rather than from search.

Rules:
- Use only `instance.context` / `instance.question`; never import common.validator.
- End with a line the official grammar can parse, e.g. "Final action: (move a b)".
  Check with `python -c "from common.validator import parse_action; print(parse_action('...'))"`.
- Tag your calls (tag="structured") so token accounting can separate them.
"""

from common.interfaces import Instance
from common.llm import Session


def solve(instance: Instance, llm: Session, temperature: float = 0.0) -> str:
    raise NotImplementedError("Faheema: structured prompting")
