
import unittest

from experiments.compare import mcnemar_exact


class TestMcNemarExact(unittest.TestCase):
    def test_six_vs_ten(self):
        self.assertAlmostEqual(mcnemar_exact(6, 10), 0.4545, places=4)

    def test_zero_vs_five(self):
        self.assertAlmostEqual(mcnemar_exact(0, 5), 0.0625, places=4)

    def test_three_vs_twelve(self):
        self.assertAlmostEqual(mcnemar_exact(3, 12), 0.0352, places=4)

    def test_equal_discordant_counts(self):
        self.assertEqual(mcnemar_exact(10, 10), 1.0)


if __name__ == "__main__":
    unittest.main()
