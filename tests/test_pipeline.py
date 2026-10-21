"""Offline checks that the shared pipeline is faithful to lm-eval. No API key needed.

    python -m unittest -v
"""

import glob
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common import validator
from common.data import load_docs, official_messages, to_instance
from common.interfaces import Budget, BudgetExceeded
from common.llm import REPO_ROOT, LLMClient
from experiments import run

DOCS = load_docs()


class FakeClient(LLMClient):
    """LLMClient whose server always answers with `reply(payload)`; counts real requests."""

    def __init__(self, reply, cache_path=None):
        super().__init__(model="fake", base_url="http://fake", cache_path=cache_path)
        self.reply = reply
        self.requests = 0

    def _post(self, payload):
        self.requests += 1
        return {"text": self.reply(payload), "prompt_tokens": 100, "completion_tokens": 10}


class TestPrompt(unittest.TestCase):
    def test_messages_identical_to_lm_eval(self):
        """Compare with the request lm-eval actually sent in the Oct 4 smoke test."""
        path = glob.glob(str(REPO_ROOT / "results/groq_smoke3/*/samples_acp_nexta_gen_*.jsonl"))[0]
        for line in Path(path).read_text(encoding="utf8").splitlines():
            sample = json.loads(line)
            sent = json.loads(sample["arguments"]["gen_args_0"]["arg_0"][0])
            self.assertEqual(official_messages(to_instance(sample["doc"])), sent)

    def test_instance_hides_answer_and_pddl(self):
        fields = set(vars(to_instance(DOCS[0])))
        self.assertEqual(fields, {"id", "domain", "context", "question"})


class TestValidator(unittest.TestCase):
    def test_known_answers(self):
        docs = [d for d in DOCS if d["answer"]["maybe"] and d["answer"]["no"]][:3]
        for doc in docs:
            yes, no = doc["answer"]["yes"][0], doc["answer"]["no"][0]
            self.assertEqual(validator.score(doc, yes).correct, True)
            self.assertEqual(validator.score(doc, f"The next action is {yes}.").correct, True)
            self.assertEqual(validator.score(doc, no).error, "not_closer")
            self.assertEqual(validator.score(doc, "I am not sure.").error, "format")
            self.assertEqual(validator.score(doc, "").error, "format")
            self.assertEqual(validator.score(doc, "(teleport everything home)").error, "inapplicable")

    def test_planner_path_runs_on_windows(self):
        """'maybe' answers need the K* planner; this crashed on Windows before the fix."""
        doc = next(d for d in DOCS if d["answer"]["maybe"])
        verdict = validator.score(doc, doc["answer"]["maybe"][0])
        self.assertIn(verdict.error, (None, "not_closer"))


class TestLLM(unittest.TestCase):
    def test_budget_enforced(self):
        llm = FakeClient(lambda p: "(a)", cache_path=None).session(Budget(max_calls=2))
        llm.chat([{"role": "user", "content": "x"}])
        llm.chat([{"role": "user", "content": "x"}])
        with self.assertRaises(BudgetExceeded):
            llm.chat([{"role": "user", "content": "x"}])
        self.assertEqual((llm.budget.calls, llm.budget.tokens), (2, 220))

    def test_cache_repeats_and_distinct_samples(self):
        with tempfile.TemporaryDirectory() as td:
            client = FakeClient(lambda p: f"seed {p['seed']}", cache_path=Path(td) / "c.sqlite")
            msgs = [{"role": "user", "content": "x"}]
            first = client.session()
            samples = [first.chat(msgs, temperature=0.7) for _ in range(3)]
            self.assertEqual(len(set(samples)), 3)  # best-of-N gets 3 different samples
            again = client.session()
            self.assertEqual([again.chat(msgs, temperature=0.7) for _ in range(3)], samples)
            self.assertEqual(client.requests, 3)  # the rerun came entirely from cache
            self.assertEqual(again.budget.calls, 3)  # but is still charged to the budget
            client._db.close()


class TestRunner(unittest.TestCase):
    def test_direct_end_to_end(self):
        gold = {to_instance(d).context: d["answer"]["yes"][0] for d in DOCS[:3]}

        def oracle_server(payload):  # pretends to be a perfect model
            context = payload["messages"][-1]["content"].split("**Question**: ")[1]
            return next(a for c, a in gold.items() if context.startswith(c))

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(run, "REPO_ROOT", Path(td)), \
                mock.patch.object(run.LLMClient, "from_env",
                                  lambda **kw: FakeClient(oracle_server, cache_path=None)):
            run.main(["--method", "direct", "--limit", "3", "--name", "t"])
            summary = json.loads((Path(td) / "results/runs/t/summary.json").read_text())
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["total_calls"], 3)


if __name__ == "__main__":
    unittest.main()
