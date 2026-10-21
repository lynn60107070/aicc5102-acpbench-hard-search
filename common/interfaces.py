"""The contract shared by every method and the runner.

Change this file only with the whole team's agreement: everyone builds against it.

A method is a function

    def solve(instance: Instance, llm: Session, **params) -> str

that returns text containing one action, e.g. "(move f4-0f f3-0f)". The runner parses
it with the official ACPBench grammar and scores it with the official validator.
Methods never see the PDDL or the answer key, so the validator cannot leak into them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instance:
    """Everything a method is allowed to see about one problem."""

    id: str
    domain: str  # e.g. "ferry"; for per-domain analysis and logging only
    context: str  # natural-language description: domain, state, goal, available actions
    question: str  # "What is the next action that takes us towards the goal?"


class BudgetExceeded(Exception):
    """Raised by Session.chat when a call would exceed the instance's budget."""


@dataclass
class Budget:
    """Per-instance LLM budget. None means unlimited.

    Cached calls are charged exactly like live ones, so accounting does not depend
    on what happens to be in the cache.
    """

    max_calls: int | None = None
    max_tokens: int | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls_by_tag: dict[str, int] = field(default_factory=dict)
    tokens_by_tag: dict[str, int] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def remaining_calls(self) -> int | None:
        return None if self.max_calls is None else self.max_calls - self.calls

    def check(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget {self.max_calls} used up")
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"token budget {self.max_tokens} used up")

    def charge(self, prompt_tokens: int, completion_tokens: int, tag: str) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls_by_tag[tag] = self.calls_by_tag.get(tag, 0) + 1
        self.tokens_by_tag[tag] = self.tokens_by_tag.get(tag, 0) + prompt_tokens + completion_tokens
