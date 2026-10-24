"""Structured prompting control (owner: Faheema).

Ask the model to write out the current state, the goal, the candidate action and the
progress it makes before giving its final action. This tests whether gains come from
extra reasoning alone rather than from search.

Rules:
- Use only `instance.context` / `instance.question`; never import common.validator.
- End with a line the official grammar can parse, e.g. "Final action: (move a b)".
  Check with `python -c "from common.validator import parse_action; print(parse_action('...'))"`.
- Tag your calls (tag="structured") so token accounting can separate them.
"""

"""Structured prompting control (owner: Faheema).

Ask the model to write out the current state, the goal, the candidate action and the
progress it makes before giving its final action. This tests whether gains come from
extra reasoning alone rather than from search.

Rules:
- End with a line the official grammar can parse, e.g. "Final action: (move a b)".
  Check with `python -c "from common.validator import parse_action; print(parse_action('...'))"`.
- Tag your calls (tag="structured") so token accounting can separate them.
"""

from common.data import OFFICIAL_MAX_TOKENS, official_messages
from common.interfaces import Instance
from common.llm import Session

def solve(instance: Instance, llm: Session, temperature: float = 0.0) -> str:
    base_msgs = official_messages(instance)
    
    structured_instruction = (
        f"\nPlease solve this by reasoning step by step. You must write out your reasoning before giving the final answer:\n"
        f"1. State: Summarize the current state based on the context.\n"
        f"2. Goal: State the objective to be achieved.\n"
        f"3. Candidate action: Propose a potential next action.\n"
        f"4. Progress: Explain how this action makes progress toward the goal.\n"
        f"5. Final action: Provide the final action in the exact format: Final action: (your action here)"
    )
    
    messages = [
        {"role": msg["role"], "content": msg["content"] + (structured_instruction if msg["role"] == "user" else "")}
        for msg in base_msgs
    ]

    response = llm.chat(
        messages,
        temperature=temperature,
        max_tokens=OFFICIAL_MAX_TOKENS,
        stop=None, # Allow full multi-step reasoning trace without early truncation
        tag="structured"
    )
    return response



