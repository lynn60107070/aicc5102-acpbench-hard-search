import tempfile
from pathlib import Path

from experiments.analyze import (
    compare_direct_to_search,
    write_reproduction_table,
)
import unittest

from experiments.analyze import calculate_auc


class TestCalculateAUC(unittest.TestCase):
    def test_perfect_ranking(self):
        scores_and_labels = [
            (0.9, True),
            (0.8, True),
            (0.2, False),
            (0.1, False),
        ]
        self.assertEqual(calculate_auc(scores_and_labels), 1.0)

    def test_tied_scores(self):
        scores_and_labels = [
            (0.8, True),
            (0.8, False),
        ]
        self.assertEqual(calculate_auc(scores_and_labels), 0.5)

    def test_missing_class_returns_none(self):
        self.assertIsNone(calculate_auc([(0.9, True), (0.7, True)]))

class TestDirectSearchComparison(unittest.TestCase):
    def test_identical_runs_have_p_value_one(self):
        rows = [
            {"id": "q1", "correct": True},
            {"id": "q2", "correct": False},
            {"id": "q3", "correct": True},
        ]

        result = compare_direct_to_search(rows, rows)

        self.assertEqual(result["direct_accuracy"], 2 / 3)
        self.assertEqual(result["search_accuracy"], 2 / 3)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["n_questions"], 3)

    def test_mismatched_question_ids_raise(self):
        direct = [{"id": "q1", "correct": True}]
        search = [{"id": "q2", "correct": True}]

        with self.assertRaises(ValueError):
            compare_direct_to_search(direct, search)


class TestReproductionTable(unittest.TestCase):
    def test_missing_runs_are_marked_pending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_reproduction_table({}, output_dir)

            content = (output_dir / "reproduction.tex").read_text(
                encoding="utf-8"
            )

            self.assertIn("Paper accuracy", content)
            self.assertIn("0.250", content)
            self.assertIn("--", content)
            self.assertIn(r"\toprule", content)
            self.assertIn(r"\bottomrule", content)
if __name__ == "__main__":
    unittest.main()
