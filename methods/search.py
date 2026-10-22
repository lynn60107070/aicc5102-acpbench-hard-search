"""LLM-guided best-first search over LLM-predicted states (owner: Lynn). The extension.

One search per instance:
    root = the state described in the problem
    repeat until no node can be expanded (depth limit d) or the call budget runs out:
        node    = the highest-scored unexpanded node with depth < d
        propose = the LLM lists k candidate actions for node's state          (1 call)
        predict = for each action, the LLM writes the resulting state          (k calls)
        score   = for each predicted state, the LLM rates progress 0-10       (k calls)
    answer = first action on the path to the highest-scored node

Only LLM-predicted states are used: no planner, simulator or applicability checker.
Each expansion costs 1 + 2k calls. Without a budget, k=3, d=2 expands the root and all 3 children
(28 calls); the call budget (--max-calls) is what makes best-first order matter: with 14 calls it
expands the root and only the most promising child.

Ablation switches (Aisha's sweeps call these by name, so keep them stable):
    k, d
    scoring        "llm" | "random" | "fixed"   fixed = trust the proposer's order (no scoring calls)
    predict_state  True | False                 False = children keep the parent's state and the
                                                LLM scores the action itself; the search is then depth 1
    temperature    for the proposal call (prediction and scoring are greedy)
"""

import random
import re
from dataclasses import dataclass

from common.interfaces import BudgetExceeded, Instance
from common.llm import Session
from common.validator import parse_actions


@dataclass
class Node:
    state: str | None  # None = the state as described in the problem
    path: list[str]  # actions applied from the root
    score: float = 0.0
    expanded: bool = False

    @property
    def depth(self) -> int:
        return len(self.path)


def _state_text(node: Node) -> str:
    if node.state is None:
        return "The current state is the one described above."
    return ("Ignore the current state given in the description above. "
            f"After applying {' '.join(node.path)}, the current state is:\n{node.state}")


def _ask(llm: Session, prompt: str, tag: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    return llm.chat([{"role": "user", "content": prompt}], temperature=temperature,
                    max_tokens=max_tokens, tag=tag)


def _propose(llm: Session, instance: Instance, node: Node, k: int, temperature: float) -> list[str]:
    prompt = (f"{instance.context}\n\n{_state_text(node)}\n\n"
              f"List {k} different actions that can be applied in the current state and that could "
              f"move us towards the goal, most promising first. Write each action on its own line, "
              f"in parentheses with all its parameters, like the available actions above. "
              f"Write only the actions.")
    return parse_actions(_ask(llm, prompt, "propose", temperature, max_tokens=200))[:k]


def _predict(llm: Session, instance: Instance, node: Node, action: str) -> str:
    prompt = (f"{instance.context}\n\n{_state_text(node)}\n\n"
              f"Now the action {action} is applied. Describe the complete resulting state: state every "
              f"fact that is true afterwards, in the same style as the description above. "
              f"Write only the state.")
    return _ask(llm, prompt, "predict").strip()


def _parse_score(text: str) -> float:
    match = re.search(r"\d+(?:\.\d+)?", text)
    return min(max(float(match.group()), 0.0), 10.0) if match else 0.0


def _score_state(llm: Session, instance: Instance, child: Node) -> float:
    prompt = (f"{instance.context}\n\n{_state_text(child)}\n\n"
              f"How close is this state to the goal? Answer with a single number from "
              f"0 (no progress) to 10 (goal reached).")
    return _parse_score(_ask(llm, prompt, "score", max_tokens=10))


def _score_action(llm: Session, instance: Instance, node: Node, action: str) -> float:
    prompt = (f"{instance.context}\n\n{_state_text(node)}\n\n"
              f"How good is applying {action} next for reaching the goal? Answer with a single number "
              f"from 0 (not applicable or useless) to 10 (certainly the right next action).")
    return _parse_score(_ask(llm, prompt, "score", max_tokens=10))


def solve(instance: Instance, llm: Session, k: int = 3, d: int = 2,
          scoring: str = "llm", predict_state: bool = True, temperature: float = 0.7) -> str:
    if scoring not in ("llm", "random", "fixed"):
        raise ValueError(f"unknown scoring {scoring!r}")
    rng = random.Random(f"{llm.client.seed}-{instance.id}")
    root = Node(state=None, path=[])
    nodes = [root]
    try:
        while True:
            frontier = [n for n in nodes
                        if not n.expanded and n.depth < d and (n is root or predict_state)]
            if not frontier:
                break
            node = max(frontier, key=lambda n: n.score)  # ties: earliest proposed wins
            node.expanded = True
            for i, action in enumerate(_propose(llm, instance, node, k, temperature)):
                state = _predict(llm, instance, node, action) if predict_state else node.state
                child = Node(state=state, path=node.path + [action])
                if scoring == "fixed":
                    child.score = node.score + k - i
                elif scoring == "random":
                    child.score = rng.random()
                elif predict_state:
                    child.score = _score_state(llm, instance, child)
                else:
                    child.score = _score_action(llm, instance, node, action)
                nodes.append(child)
    except BudgetExceeded:
        pass  # answer with the best node found so far
    candidates = [n for n in nodes if n.path]
    if not candidates:
        return ""  # nothing proposed: scored as a format error
    return max(candidates, key=lambda n: n.score).path[0]
