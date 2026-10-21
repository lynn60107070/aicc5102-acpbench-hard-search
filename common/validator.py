"""Scoring with the official ACPBench-Hard grammar and validator, plus error labels.

Only the runner may import this module. Methods must never call it: that would be
oracle leakage.

Correctness is exactly the official NextActionEvaluator's. On top of it we label why
an answer is wrong:
    format        the official grammar could not parse an action from the response
    inapplicable  the action cannot be applied in the current state
    not_closer    applicable, but does not reduce the optimal distance to the goal by 1
"""

import functools
import os
import tempfile
from dataclasses import dataclass

# Windows fix: the official validator writes to NamedTemporaryFile(...).name while the
# file is still open, which raises PermissionError on Windows for every "maybe" answer.
# delete_on_close=False (Python 3.12+) keeps automatic deletion but allows reopening.
if os.name == "nt":
    tempfile.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete_on_close=False)

import lm_eval.tasks.acpbench.gen_2shot.acp_utils as acp  # noqa: E402

_parser = acp.ACPGrammarParser("action_name")


@dataclass
class Verdict:
    correct: bool
    error: str | None  # None when correct, else "format" | "inapplicable" | "not_closer"
    action: str | None  # the action the grammar extracted, or None


def parse_action(response: str) -> str | None:
    """Extract the action exactly as lm-eval's ACP_grammar_filter does."""
    parsed = _parser.parse(response)
    return None if parsed in (None, "None") else parsed


def score(doc: dict, response: str) -> Verdict:
    action = parse_action(response)
    if action is None:
        return Verdict(False, "format", None)
    correct = bool(acp.NextActionEvaluator().get_score([action], doc))
    if correct:
        return Verdict(True, None, action)
    a = action.strip().lower()
    answer = doc["answer"]
    known = {x.lower() for x in answer["yes"] + answer["no"] + answer["maybe"]}
    return Verdict(False, "not_closer" if a in known else "inapplicable", action)
