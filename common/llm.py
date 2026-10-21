"""OpenAI-compatible chat client with a disk cache and per-instance budget accounting.

Works with any OpenAI-compatible endpoint: Groq, Together, OpenRouter, vLLM, Ollama.

    client = LLMClient.from_env()            # reads .env / environment variables
    llm = client.session(Budget(max_calls=20))
    text = llm.chat(messages, temperature=0.7, tag="propose")

Caching: every response is stored in SQLite, keyed by the full request plus the run seed
plus how many times this exact request was already made in the session. Re-running an
experiment therefore costs nothing and returns identical outputs, while best-of-N's N
identical sampled requests still get N different responses.
"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import requests

from common.interfaces import Budget

REPO_ROOT = Path(__file__).resolve().parent.parent
# Outside OneDrive on purpose: a syncing SQLite file gets locked.
DEFAULT_CACHE = Path.home() / ".cache" / "aicc5102_acp" / "llm_cache.sqlite"
LM_EVAL_SEED = 1234  # the seed lm-eval sends; keeps direct generation identical to the harness


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines); real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMClient:
    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 seed: int = LM_EVAL_SEED, cache_path: Path | None = DEFAULT_CACHE,
                 timeout: float = 180, max_retries: int = 8, provider: str | None = None):
        self.model = model
        # OpenRouter only: pin one backend so every request runs at the same precision.
        self.provider = provider
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            base_url += "/chat/completions"
        self.url = base_url
        self.api_key = api_key
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        self._db = None
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
            self._db.commit()

    @classmethod
    def from_env(cls, **overrides) -> "LLMClient":
        load_dotenv()
        kwargs = dict(
            model=os.environ.get("ACP_MODEL"),
            base_url=os.environ.get("ACP_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            provider=os.environ.get("ACP_PROVIDER") or None,
        )
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        if not kwargs["model"] or not kwargs["base_url"]:
            raise SystemExit("Set ACP_MODEL and ACP_BASE_URL (in .env or pass --model / --base-url).")
        return cls(**kwargs)

    def session(self, budget: Budget | None = None) -> "Session":
        return Session(self, budget or Budget())

    def complete(self, payload: dict, occurrence: int) -> dict:
        """Return {"text", "prompt_tokens", "completion_tokens"}, from cache if possible."""
        key = hashlib.sha256(json.dumps(
            {"url": self.url, "seed": self.seed, "occurrence": occurrence, **payload},
            sort_keys=True).encode()).hexdigest()
        if self._db is not None:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
            if row:
                return json.loads(row[0])
        result = self._post(payload)
        if self._db is not None:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(result)))
            self._db.commit()
        return result

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage") or {}
                return {
                    "text": text,
                    # Rough 4-chars-per-token estimate only if the server reports no usage.
                    "prompt_tokens": usage.get("prompt_tokens",
                                               sum(len(m["content"]) for m in payload["messages"]) // 4),
                    "completion_tokens": usage.get("completion_tokens", len(text) // 4),
                    # Which backend served it (OpenRouter routes between providers): fidelity log D1.
                    "provider": data.get("provider"),
                }
            if r is not None and r.status_code not in (408, 409, 429) and r.status_code < 500:
                raise RuntimeError(f"LLM request failed {r.status_code}: {r.text[:500]}")
            retry_after = r.headers.get("retry-after") if r is not None else None
            wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
            time.sleep(wait)
        raise RuntimeError(f"LLM request failed after {self.max_retries} retries")


class Session:
    """One instance's view of the LLM: enforces and records its budget."""

    def __init__(self, client: LLMClient, budget: Budget):
        self.client = client
        self.budget = budget
        self.log: list[dict] = []  # every call this instance made, for debugging and analysis
        self._seen: dict[str, int] = {}

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1000,
             stop: list[str] | None = None, tag: str = "default") -> str:
        """Send one chat request and return the reply text. Raises BudgetExceeded."""
        self.budget.check()
        payload = {"model": self.client.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if self.client.provider:
            payload["provider"] = {"order": [self.client.provider], "allow_fallbacks": False}
        if stop:
            payload["stop"] = stop[:4]  # the OpenAI API accepts at most 4 stop sequences
        request_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        occurrence = self._seen.get(request_id, 0)
        self._seen[request_id] = occurrence + 1
        # Sampled repeats of the same request need different server seeds to differ.
        payload["seed"] = self.client.seed + occurrence
        result = self.client.complete(payload, occurrence)
        self.budget.charge(result["prompt_tokens"], result["completion_tokens"], tag)
        self.log.append({"tag": tag, "temperature": temperature, "messages": messages, **result})
        return result["text"]
