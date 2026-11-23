"""Offline tests for the follow-up statistics (experiments/followups.py) and structured + vote."""

import unittest

from common.data import load_docs, to_instance
from common.interfaces import Budget
from experiments import followups
from methods import structured_vote
from tests.test_pipeline import FakeClient


class TestFollowupStats(unittest.TestCase):
    def test_binomial_two_sided(self):
        self.assertAlmostEqual(followups.binomial_two_sided(5, 10, 0.5), 1.0)
        self.assertAlmostEqual(followups.binomial_two_sided(2, 10, 0.5), followups.binomial_two_sided(8, 10, 0.5))
        self.assertLess(followups.binomial_two_sided(10, 10, 0.5), 0.01)

    def test_pooled_test_identical_runs(self):
        rows = [[{"id": str(i), "correct": i % 2 == 0} for i in range(20)]]
        result = followups.pooled_test(rows, rows, n_perm=200, n_boot=200)
        self.assertEqual(result["diff"], 0)
        self.assertEqual(result["p"], 1.0)

    def test_pooled_test_clear_difference(self):
        better = [[{"id": str(i), "correct": True} for i in range(30)]]
        worse = [[{"id": str(i), "correct": False} for i in range(30)]]
        result = followups.pooled_test(better, worse, n_perm=500, n_boot=200)
        self.assertEqual(result["diff"], 1)
        self.assertLess(result["p"], 0.01)
        self.assertGreater(result["ci95"][0], 0)


class TestStructuredVote(unittest.TestCase):
    def test_returns_majority_final_action(self):
        replies = iter(["reasoning\nFinal action: (a)", "Final action: (b)", "Final action: (a)"])
        llm = FakeClient(lambda p: next(replies), cache_path=None).session(Budget(max_calls=3))
        instance = to_instance(load_docs()[0])
        self.assertEqual(structured_vote.solve(instance, llm, n=3), "(a)")

    def test_budget_exhaustion_votes_over_collected(self):
        llm = FakeClient(lambda p: "Final action: (c)", cache_path=None).session(Budget(max_calls=2))
        instance = to_instance(load_docs()[0])
        self.assertEqual(structured_vote.solve(instance, llm, n=14), "(c)")
        self.assertTrue(llm.budget.exhausted)


if __name__ == "__main__":
    unittest.main()
