"""Offline checks of the search mechanics with a scripted fake LLM.

    python -m unittest tests.test_search -v
"""

import unittest

from common.interfaces import Budget, Instance
from methods import search
from tests.test_pipeline import FakeClient

INSTANCE = Instance(id="1", domain="toy", context="Toy problem.", question="Next action?")


def scripted(payload):
    """Proposals: (a)(b)(c) at the root, (x)(y)(z) below. State scores: path via (b) looks best."""
    prompt = payload["messages"][-1]["content"]
    if "List " in prompt:
        return "(x)\n(y)\n(z)" if "After applying" in prompt else "(a)\n(b)\n(c)"
    if "resulting state" in prompt:
        return "state after " + prompt.split("applying the action ")[1].split(". First")[0]
    if "How close" in prompt:
        return "Reasoning.\nScore: 9" if "state after (b)" in prompt else "Reasoning.\nScore: 3"
    if "How good" in prompt:
        return "Score: 8" if "(c)" in prompt else "Score: 2"
    raise AssertionError(prompt)


def run(budget=None, **params):
    llm = FakeClient(scripted, cache_path=None).session(budget or Budget())
    return search.solve(INSTANCE, llm, **params), llm.budget


class TestSearch(unittest.TestCase):
    def test_llm_scoring_picks_best_state(self):
        answer, budget = run(k=3, d=2)
        self.assertEqual(answer, "(b)")
        self.assertEqual(budget.calls, 28)  # unbudgeted: root + all 3 children, 7 calls each

    def test_budget_expands_best_child_first(self):
        llm = FakeClient(scripted, cache_path=None).session(Budget(max_calls=14))
        answer = search.solve(INSTANCE, llm, k=3, d=2)
        self.assertEqual(answer, "(b)")
        self.assertEqual(llm.budget.calls_by_tag, {"propose": 2, "predict": 6, "score": 6})
        second_proposal = [c for c in llm.log if c["tag"] == "propose"][1]
        self.assertIn("After applying (b)", second_proposal["messages"][-1]["content"])

    def test_fixed_scoring_trusts_proposer(self):
        answer, budget = run(k=3, d=2, scoring="fixed")
        self.assertEqual(answer, "(a)")
        self.assertNotIn("score", budget.calls_by_tag)

    def test_without_state_prediction(self):
        answer, budget = run(k=3, d=2, predict_state=False)
        self.assertEqual(answer, "(c)")
        self.assertEqual(budget.calls, 4)  # depth 1 only: 1 propose + 3 action scores

    def test_budget_returns_best_so_far(self):
        answer, budget = run(Budget(max_calls=5), k=3, d=2)
        self.assertEqual(budget.calls, 5)  # propose, predict a, score a, predict b, score b
        self.assertEqual(answer, "(b)")

    def test_inapplicable_prediction_is_pruned(self):
        def b_inapplicable(payload):
            prompt = payload["messages"][-1]["content"]
            if "resulting state" in prompt and "applying the action (b)" in prompt:
                return "NOT APPLICABLE"
            return scripted(payload)

        llm = FakeClient(b_inapplicable, cache_path=None).session(Budget(max_calls=6))
        answer = search.solve(INSTANCE, llm, k=3, d=2)
        self.assertNotEqual(answer, "(b)")  # (b) would have scored 9, but the LLM ruled it out
        self.assertEqual(llm.budget.calls_by_tag, {"propose": 1, "predict": 3, "score": 2})

    def test_score_parsing(self):
        self.assertEqual(search._parse_score("## Step 1: Determine the number of cars"), 0.0)
        self.assertEqual(search._parse_score("The ferry is close.\nScore: 7"), 7.0)
        self.assertEqual(search._parse_score("**Score:** 6.5"), 6.5)
        self.assertEqual(search._parse_score("Score: 3 at first... Score: 8"), 8.0)
        self.assertEqual(search._parse_score("Score: 15"), 10.0)

    def test_random_scoring_is_reproducible(self):
        self.assertEqual(run(k=3, d=2, scoring="random")[0], run(k=3, d=2, scoring="random")[0])


if __name__ == "__main__":
    unittest.main()
