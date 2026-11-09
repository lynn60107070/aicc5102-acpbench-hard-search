"""Offline checks of the baselines with a scripted fake LLM.

    python -m unittest tests.test_baselines -v
"""

import unittest

from common.interfaces import Budget, Instance
from methods import best_of_n, structured, vote_check
from tests.test_pipeline import FakeClient

INSTANCE = Instance(id="1", domain="toy", context="Toy problem.", question="Next action?")


def scripted(replies):
    """Fake model: returns the given replies one after another."""
    calls = []

    def reply(payload):
        calls.append(payload)
        return replies[len(calls) - 1]

    return reply


class TestBaselines(unittest.TestCase):
    def test_best_of_n_returns_majority_action(self):
        fake = scripted(["(sail l1 l0)", "(board c1 l1)", "(sail l1 l0)"])
        llm = FakeClient(fake, cache_path=None).session(Budget())
        answer = best_of_n.solve(INSTANCE, llm, n=3)
        self.assertEqual(answer, "(sail l1 l0)")

    def test_structured_returns_action_after_final_action(self):
        fake = scripted(["State: ...\nCandidate action: (board c1 l1)\nFinal action: (sail l1 l0)"])
        llm = FakeClient(fake, cache_path=None).session(Budget())
        answer = structured.solve(INSTANCE, llm)
        self.assertEqual(answer.strip(), "(sail l1 l0)")

    def test_vote_check_skips_rejected_top_vote(self):
        fake = scripted(["(sail l1 l0)", "(board c1 l1)", "(sail l1 l0)",
                         "1. fails\nApplicable: no", "All conditions hold, so it can be applied."])
        llm = FakeClient(fake, cache_path=None).session(Budget())
        answer = vote_check.solve(INSTANCE, llm, n=3, m=2)
        self.assertEqual(answer, "(board c1 l1)")

    def test_vote_check_falls_back_to_top_vote(self):
        fake = scripted(["(sail l1 l0)", "(board c1 l1)", "(sail l1 l0)",
                         "Applicable: no", "It is not applicable."])
        llm = FakeClient(fake, cache_path=None).session(Budget())
        answer = vote_check.solve(INSTANCE, llm, n=3, m=2)
        self.assertEqual(answer, "(sail l1 l0)")


if __name__ == "__main__":
    unittest.main()