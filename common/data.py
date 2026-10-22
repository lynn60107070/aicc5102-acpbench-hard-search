"""ACPBench-Hard Next Action data and the official 2-shot prompt.

The few-shot examples and prompt template are read from the installed lm-eval task
config, so our direct-generation prompt is byte-identical to the harness's.
"""

import re
from pathlib import Path

import yaml
from datasets import load_dataset

import lm_eval.tasks.acpbench.gen_2shot.acp_utils as _acp_utils
from common.interfaces import Instance

DATASET = "ibm-research/acp_bench"
CONFIG = "acp_nexta_gen"
SPLIT = "test"

# lm-eval sends the first 4 of the task's `until` list (OpenAI API limit).
OFFICIAL_STOP = ["\n\n\n\n", "\n\n", "**Question**:", "**Question:**"]
OFFICIAL_MAX_TOKENS = 1000

_TASK_DIR = Path(_acp_utils.__file__).parent
_FEWSHOT = yaml.safe_load((_TASK_DIR / "next_act.yaml").read_text(encoding="utf8"))["fewshot_config"]["samples"]


def load_docs(limit: int | None = None, ids: list[str] | None = None, split: str = SPLIT) -> list[dict]:
    """Raw dataset rows, including PDDL and answer key. Only the runner and validator use these.

    split="validation" has 40 questions (8 domains, no overlap with test): tune methods there,
    report on "test" (130), so tuning never sees the reported questions.
    """
    docs = list(load_dataset(DATASET, CONFIG, split=split))
    if ids is not None:
        wanted = set(ids)
        docs = [d for d in docs if str(d["id"]) in wanted]
    return docs[:limit] if limit else docs


def domain_of(doc: dict) -> str:
    match = re.search(r"\(domain\s+([^\s)]+)", doc["PDDL_domain"])
    return match.group(1) if match else "unknown"


def to_instance(doc: dict) -> Instance:
    """Strip a raw row down to what methods may see."""
    return Instance(id=str(doc["id"]), domain=domain_of(doc),
                    context=doc["context"], question=doc["question"])


def official_question(context: str, question: str) -> str:
    return (f"**Question**: {context} {question} Each action starts with an opening parenthesis "
            f"and ends with closing parenthesis. Provide only the action. **Final Answer**:")


def official_messages(instance: Instance) -> list[dict]:
    """The exact chat messages lm-eval sends for acp_nexta_gen with --apply_chat_template."""
    messages = []
    for shot in _FEWSHOT:
        messages.append({"role": "user", "content": official_question(shot["context"], shot["question"])})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": official_question(instance.context, instance.question)})
    return messages
