"""LLM-guided best-first search over LLM-predicted states (owner: Lynn). The extension.

Plan:
    1. propose: ask the LLM for k candidate actions from the current (predicted) state
    2. predict: ask the LLM for the state each action leads to
    3. score:   ask the LLM how promising each predicted state is
    4. expand the best-scored frontier state, until depth d or the call budget runs out
    5. return the first action of the best path

Only LLM-predicted states are used: no planner, simulator or applicability checker.
Never import common.validator here, except `parse_action` for reading the LLM's output.

Ablation switches (Aisha's sweeps call these by name, so keep them stable):
    k              candidates per expansion
    d              depth limit
    scoring        "llm" | "random" | "fixed"   (fixed = expand in proposal order)
    predict_state  True | False                 (False = score actions on the root state)

Catch BudgetExceeded and return the best action found so far.
"""

from common.interfaces import Instance
from common.llm import Session


def solve(instance: Instance, llm: Session, k: int = 3, d: int = 2,
          scoring: str = "llm", predict_state: bool = True, temperature: float = 0.7) -> str:
    raise NotImplementedError("Lynn: best-first search")
