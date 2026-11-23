"""Scoring with the official ACPBench-Hard grammar and validator, plus error labels.

Only the runner may call `score`. Methods must never call it: that would be oracle
leakage. Methods may use `parse_action` / `parse_actions`, which apply only the output
grammar and see no PDDL or answer key.

Correctness is exactly the official NextActionEvaluator's. On top of it we label why
an answer is wrong:
    format        the official grammar could not parse an action from the response
    inapplicable  the action cannot be applied in the current state
    not_closer    applicable, but does not reduce the optimal distance to the goal by 1
"""

import atexit
import functools
import os
import shutil
import tempfile
from dataclasses import dataclass

# Windows fix: the official validator writes to NamedTemporaryFile(...).name while the
# file is still open, which raises PermissionError on Windows for every "maybe" answer.
# delete_on_close=False (Python 3.12+) keeps automatic deletion but allows reopening.
if os.name == "nt":
    tempfile.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete_on_close=False)

import lm_eval.tasks.acpbench.gen_2shot.acp_utils as acp  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402

import kstar_planner.planners as _kstar  # noqa: E402

# The official planner (K*, run by the validator for "maybe" answers) writes its intermediate
# output.sas into the current directory and deletes it afterwards. Two processes scoring from
# the same directory at once can read each other's file and score an answer against the wrong
# problem. Run every planner call of this process in its own private directory instead.
_PLANNER_DIR = tempfile.mkdtemp(prefix="acp_planner_")
atexit.register(shutil.rmtree, _PLANNER_DIR, ignore_errors=True)


def _run_in_private_dir(*args, **kwargs):
    kwargs.setdefault("cwd", _PLANNER_DIR)
    return subprocess.run(*args, **kwargs)


_kstar.subprocess = types.SimpleNamespace(run=_run_in_private_dir, PIPE=subprocess.PIPE)

_parser = acp.ACPGrammarParser("action_name")
_list_parser = acp.ACPGrammarParser("action_list")
MAYBE_RETRIES = 2  # extra checks of a 'maybe' answer that came back wrong (see score)


@dataclass
class Verdict:
    correct: bool
    error: str | None  # None when correct, else "format" | "inapplicable" | "not_closer"
    action: str | None  # the action the grammar extracted, or None


def parse_action(response: str) -> str | None:
    """Extract the action exactly as lm-eval's ACP_grammar_filter does."""
    parsed = _parser.parse(response)
    return None if parsed in (None, "None") else parsed


def parse_actions(response: str) -> list[str]:
    """Extract every action from a list-style response, with the official grammar. Order kept, duplicates dropped."""
    parsed = _list_parser.parse(response) or []
    return list(dict.fromkeys(a.strip().lower() for a in parsed))


_scores: dict[tuple[str, str], bool] = {}  # (question id, action) -> correct; analysis re-scores the same pairs


def score(doc: dict, response: str) -> Verdict:
    action = parse_action(response)
    if action is None:
        return Verdict(False, "format", None)
    key = (str(doc["id"]), action)
    if key not in _scores:
        _scores[key] = _official_correct(doc, action)
    correct = _scores[key]
    if correct:
        return Verdict(True, None, action)
    a = action.strip().lower()
    answer = doc["answer"]
    known = {x.lower() for x in answer["yes"] + answer["no"] + answer["maybe"]}
    return Verdict(False, "not_closer" if a in known else "inapplicable", action)


def _official_correct(doc: dict, action: str) -> bool:
    correct = bool(acp.NextActionEvaluator().get_score([action], doc))
    a = action.strip().lower()
    if not correct and a in {x.lower() for x in doc["answer"]["maybe"]}:
        # "maybe" actions are decided by a planner with a 5 s timeout, and a timeout scores the
        # answer wrong, so on a busy machine a correct answer can come back wrong (never the
        # reverse). Re-check twice; any success means the action is correct.
        correct = any(bool(acp.NextActionEvaluator().get_score([action], doc)) for _ in range(MAYBE_RETRIES))
    return correct
