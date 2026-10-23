"""LLM-guided best-first search over LLM-predicted states (owner: Lynn). The extension.

One search per instance:
    root = the state described in the problem
    repeat until no node can be expanded (depth limit d) or the call budget runs out:
        node    = the highest-scored unexpanded node with depth < d
        propose = the LLM lists k candidate actions for node's state          (1 call)
        predict = for each action, the LLM writes the resulting state, or says (k calls)
                  NOT APPLICABLE; such actions are pruned and not scored
        score   = for each predicted state, the LLM reasons briefly and       (≤ k calls)
                  ends with "Score: N" (0-10)
    answer = first action on the path to the highest-scored node

Only LLM-predicted states are used: no planner, simulator or applicability checker.
Each expansion costs 1 + 2k calls. Without a budget, k=3, d=2 expands the root and all 3 children
(28 calls); the call budget (--max-calls) is what makes best-first order matter: with 14 calls it
expands the root and only the most promising child.

Ablation switches (Aisha's sweeps call these by name, so keep them stable):
    k, d
    scoring        "llm"    0-10 closeness score per state (v1; barely discriminates for 8B)
                   "steps"  A*-style: LLM estimates remaining actions h, rank by f = depth + h
                   "vote"   one comparative call per expansion picks the best child (as in ToT)
                   "random" | "fixed"           fixed = trust the proposer's order (no scoring calls)
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
              f"in parentheses, with the action name followed only by object names from the problem "
              f"(no other words), like the available actions above. Write only the actions.")
    return parse_actions(_ask(llm, prompt, "propose", temperature, max_tokens=200))[:k]


NOT_APPLICABLE = "NOT APPLICABLE"


def _predict(llm: Session, instance: Instance, node: Node, action: str) -> str | None:
    """The predicted next state, or None if the LLM judges the action inapplicable."""
    prompt = (f"{instance.context}\n\n{_state_text(node)}\n\n"
              f"Consider applying the action {action}. First check whether it can be applied in the "
              f"current state. If it cannot, reply only: {NOT_APPLICABLE}. Otherwise describe the "
              f"complete resulting state: state every fact that is true afterwards, in the same style "
              f"as the description above. Write only the state.")
    text = _ask(llm, prompt, "predict").strip()
    return None if NOT_APPLICABLE in text.upper() else text


def _parse_score(text: str) -> float:
    """The number on the last 'Score: N' line; 0 if the model gave none."""
    matches = re.findall(r"score\W{0,3}\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    return min(max(float(matches[-1]), 0.0), 10.0) if matches else 0.0


_SCORE_FORMAT = "Reason briefly in at most three sentences, then end with a final line of the form: Score: N"


def _score_state(llm: Session, instance: Instance, child: Node) -> float:
    prompt = (f"{instance.context}\n\n{_state_text(child)}\n\n"
              f"How close is this state to the goal, on a scale from 0 (no progress) to 10 "
              f"(goal reached)? {_SCORE_FORMAT}")
    return _parse_score(_ask(llm, prompt, "score", max_tokens=400))


def _score_action(llm: Session, instance: Instance, node: Node, action: str) -> float:
    prompt = (f"{instance.context}\n\n{_state_text(node)}\n\n"
              f"How good is applying {action} next for reaching the goal, on a scale from 0 (not "
              f"applicable or useless) to 10 (certainly the right next action)? {_SCORE_FORMAT}")
    return _parse_score(_ask(llm, prompt, "score", max_tokens=400))


UNKNOWN_STEPS = 99  # the LLM gave no estimate: rank such nodes last


def _estimate_steps(llm: Session, instance: Instance, node: Node, action: str | None = None) -> int:
    """LLM estimate of the remaining plan length from node's state (after `action`, if given)."""
    what = "this state" if action is None else f"the current state, if we apply {action} next"
    prompt = (f"{instance.context}\n\n{_state_text(node)}\n\n"
              f"Estimate the minimum number of actions still needed to reach the goal from {what}. "
              f"Reason briefly in at most three sentences, then end with a final line of the form: "
              f"Steps: N")
    text = _ask(llm, prompt, "score", max_tokens=400)
    matches = re.findall(r"steps\W{0,3}\s*(\d+)", text, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else UNKNOWN_STEPS


def _vote(llm: Session, instance: Instance, node: Node, children: list[Node]) -> int:
    """Index of the child the LLM judges to make the most progress (Tree-of-Thoughts-style vote)."""
    options = "\n\n".join(
        f"Option {j + 1}: apply {c.path[-1]}." + (f" Resulting state:\n{c.state}" if c.state else "")
        for j, c in enumerate(children))
    prompt = (f"{instance.context}\n\n{_state_text(node)}\n\n{options}\n\n"
              f"Which option makes the most progress towards the goal? Reason briefly in at most "
              f"three sentences, then end with a final line of the form: Best: N")
    matches = re.findall(r"best\W{0,3}\s*(?:option\s*)?(\d+)",
                         _ask(llm, prompt, "score", max_tokens=400), flags=re.IGNORECASE)
    best = int(matches[-1]) - 1 if matches else 0
    return best if 0 <= best < len(children) else 0


SCORINGS = ("llm", "steps", "vote", "random", "fixed")


def solve(instance: Instance, llm: Session, k: int = 3, d: int = 2,
          scoring: str = "llm", predict_state: bool = True, temperature: float = 0.7) -> str:
    if scoring not in SCORINGS:
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
            children = []
            for i, action in enumerate(_propose(llm, instance, node, k, temperature)):
                state = _predict(llm, instance, node, action) if predict_state else node.state
                child = Node(state=state, path=node.path + [action])
                if predict_state and state is None:
                    # The LLM itself judged the action inapplicable: never expand it, never pick it
                    # over an applicable candidate, and skip its scoring call.
                    child.score, child.expanded = float("-inf"), True
                    nodes.append(child)
                    continue
                if scoring == "fixed":
                    child.score = node.score + k - i
                elif scoring == "random":
                    child.score = rng.random()
                elif scoring == "llm":
                    child.score = (_score_state(llm, instance, child) if predict_state
                                   else _score_action(llm, instance, node, action))
                elif scoring == "steps":
                    # A*-style: f = g + h with g = depth and h = the LLM's estimate; higher score = lower f
                    h = (_estimate_steps(llm, instance, child) if predict_state
                         else _estimate_steps(llm, instance, node, action))
                    child.score = -(child.depth + h)
                # Only now join the tree: a child whose scoring call hit the budget is dropped,
                # so an unscored default can never outrank scored nodes.
                nodes.append(child)
                children.append(child)
            if scoring == "vote" and children:
                # One comparative call per expansion; the chosen child outranks its siblings.
                best = _vote(llm, instance, node, children) if len(children) > 1 else 0
                for j, child in enumerate(children):
                    child.score = node.score + (1 if j == best else 0)
    except BudgetExceeded:
        pass  # answer with the best node found so far
    candidates = [n for n in nodes if n.path]
    if not candidates:
        return ""  # nothing proposed: scored as a format error
    return max(candidates, key=lambda n: n.score).path[0]
