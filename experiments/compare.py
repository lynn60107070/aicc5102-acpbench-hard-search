
import argparse
import json
from pathlib import Path
import math

def load_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows

def index_by_id(rows):
    indexed = {}

    for row in rows:
        question_id = row["id"]

        if question_id in indexed:
            raise ValueError(f"Duplicate question ID found: {question_id}")

        indexed[question_id] = row

    return indexed

def match_questions(rows_a, rows_b):
    run_a = index_by_id(rows_a)
    run_b = index_by_id(rows_b)

    ids_a = set(run_a.keys())
    ids_b = set(run_b.keys())

    if ids_a != ids_b:
        missing_from_b = ids_a - ids_b
        missing_from_a = ids_b - ids_a

        raise ValueError(
            f"Question IDs do not match. "
            f"Missing from Run B: {missing_from_b}. "
            f"Missing from Run A: {missing_from_a}."
        )

    return run_a, run_b

def compare_correctness(run_a, run_b):
    both_correct = 0
    a_only_correct = 0
    b_only_correct = 0
    both_wrong = 0

    for question_id in run_a:
        correct_a = run_a[question_id]["correct"]
        correct_b = run_b[question_id]["correct"]

        if correct_a and correct_b:
            both_correct += 1
        elif correct_a and not correct_b:
            a_only_correct += 1
        elif not correct_a and correct_b:
            b_only_correct += 1
        else:
            both_wrong += 1

    return both_correct, a_only_correct, b_only_correct, both_wrong


def calculate_accuracy(rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)

    if total == 0:
        raise ValueError("Cannot calculate accuracy with no questions.")

    return correct / total

def mcnemar_exact(a_only_correct, b_only_correct):
    n = a_only_correct + b_only_correct

    if n == 0:
        return 1.0

    smaller = min(a_only_correct, b_only_correct)

    probability = sum(
        math.comb(n, k) * (0.5 ** n)
        for k in range(smaller + 1)
    )

    return min(1.0, 2 * probability)


def main():
    parser = argparse.ArgumentParser(
        description="Compare the accuracy of two experiment runs."
    )
    parser.add_argument("run_a", help="Path to Run A's rows.jsonl")
    parser.add_argument("run_b", help="Path to Run B's rows.jsonl")
    args = parser.parse_args()

    rows_a = load_rows(args.run_a)
    rows_b = load_rows(args.run_b)

    run_a, run_b = match_questions(rows_a, rows_b)

    accuracy_a = calculate_accuracy(list(run_a.values()))
    accuracy_b = calculate_accuracy(list(run_b.values()))

    both_correct, a_only_correct, b_only_correct, both_wrong = (
    compare_correctness(run_a, run_b))

    p_value = mcnemar_exact(a_only_correct, b_only_correct)

    print(f"Run A accuracy: {accuracy_a:.1%}")
    print(f"Run B accuracy: {accuracy_b:.1%}")
    print(f"Accuracy difference (B - A): {(accuracy_b - accuracy_a) * 100:.2f} percentage points")
    print(f"A-only correct: {a_only_correct}")
    print(f"B-only correct: {b_only_correct}")
    print(f"McNemar exact p-value: {p_value:.4f}")


if __name__ == "__main__":
    main()




