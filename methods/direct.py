"""Direct generation: the ACPBench-Hard baseline we reproduce (owner: Lynn, verified by Faheema).

Same messages, decoding settings and stop sequences as lm-eval's acp_nexta_gen task.
"""

from common.data import OFFICIAL_MAX_TOKENS, OFFICIAL_STOP, official_messages
from common.interfaces import Instance
from common.llm import Session


def solve(instance: Instance, llm: Session) -> str:
    return llm.chat(official_messages(instance), temperature=0.0,
                    max_tokens=OFFICIAL_MAX_TOKENS, stop=OFFICIAL_STOP, tag="direct")
