"""LLM client: OpenAI-compatible or native Ollama, with schema-enforced, self-healing calls.

`LLM.structured()` is the only way stages talk to a model:
  1. ask for JSON (Ollama: grammar-constrained to the Pydantic JSON schema),
  2. extract + lightly repair the JSON, validate against the Pydantic model,
  3. on failure, feed the validation errors back to the model and retry (max_retries),
  4. if still failing, raise StructuredOutputError so the caller can fall back
     deterministically instead of crashing an unattended run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import typing
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .budget import Budget, BudgetExceeded
from .config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class BackendUnavailable(RuntimeError):
    """The model server is down, out of memory or stalled.

    Deliberately NOT an LLMError: stage fallbacks must not paper over a dead backend
    (that would silently build a junk agent). The pipeline aborts cleanly instead.
    """


# Errors a stage may answer with its deterministic fallback. A dead backend is not one of them.
FALLBACK_ERRORS = (LLMError, BudgetExceeded)


class StructuredOutputError(LLMError):
    def __init__(self, msg: str, last_raw: str = "", errors: str = ""):
        super().__init__(msg)
        self.last_raw = last_raw
        self.errors = errors


# --------------------------------------------------------------------------- JSON helpers
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _balanced_span(s: str, start: int) -> Optional[str]:
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _light_repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = re.sub(r"(?<![\w\"])True(?![\w\"])", "true", s)
    s = re.sub(r"(?<![\w\"])False(?![\w\"])", "false", s)
    s = re.sub(r"(?<![\w\"])None(?![\w\"])", "null", s)
    return s


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of model output. Raises ValueError."""
    text = _THINK.sub("", text or "").strip()
    candidates = [m.group(1).strip() for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        for attempt in (cand, _light_repair(cand)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
        for i, ch in enumerate(cand):
            if ch in "{[":
                span = _balanced_span(cand, i)
                if span:
                    for attempt in (span, _light_repair(span)):
                        try:
                            return json.loads(attempt)
                        except json.JSONDecodeError:
                            pass
                break
    raise ValueError("no parseable JSON object found in model output")


def _coerce_top_level(data: Any, schema: Type[BaseModel]) -> Any:
    """Small models often return a bare list where {"items": [...]} was asked, or wrap
    the object one level deep ({"requirements": {...}}). Fix both."""
    fields = schema.model_fields
    if isinstance(data, list):
        list_fields = [n for n, f in fields.items() if typing.get_origin(f.annotation) is list]
        if len(list_fields) == 1:
            return {list_fields[0]: data}
    if isinstance(data, dict) and len(data) == 1:
        (k, v), = data.items()
        if k not in fields and isinstance(v, dict) and set(v) & set(fields):
            return v
    return data


def skeleton(model: Type[BaseModel]) -> dict:
    """A compact example object for the prompt - far easier for a 3B model than a JSON schema."""

    def render(ann: Any) -> Any:
        origin = typing.get_origin(ann)
        args = typing.get_args(ann)
        if origin in (typing.Union, getattr(__import__("types"), "UnionType", object)):
            non_none = [a for a in args if a is not type(None)]
            return render(non_none[0]) if non_none else None
        if origin is list:
            return [render(args[0]) if args else "<string>"]
        if origin is dict:
            return {"<key>": "<value>"}
        if origin is typing.Literal:
            return "|".join(map(str, args))
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return skeleton(ann)
        if isinstance(ann, type) and issubclass(ann, __import__("enum").Enum):
            return "|".join(e.value for e in ann)
        return {str: "<string>", int: 0, float: 0.0, bool: False}.get(ann, "<value>")

    return {name: render(f.annotation) for name, f in model.model_fields.items()}


def _format_errors(e: ValidationError) -> str:
    out = []
    for err in e.errors()[:8]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
        out.append(f"- {loc}: {err.get('msg')}")
    return "\n".join(out)


# --------------------------------------------------------------------------- client
class LLM:
    def __init__(self, settings: Optional[Settings] = None, budget: Optional[Budget] = None, use_cache: bool = True):
        self.s = settings or get_settings()
        self.budget = budget or Budget(self.s.max_tokens, self.s.max_seconds)
        self.use_cache = use_cache
        self.backend = self._detect_backend()
        self.num_ctx = int(os.getenv("FORGE_NUM_CTX", "8192"))
        self._cache_dir = self.s.cache_dir / "llm"
        self._embed_ok: Optional[bool] = None

    # ---------------------------------------------------------------- plumbing
    def _detect_backend(self) -> str:
        b = os.getenv("FORGE_BACKEND", "auto").lower()
        if b in ("ollama", "openai"):
            return b
        return "ollama" if ":11434" in self.s.base_url else "openai"

    @property
    def _ollama_root(self) -> str:
        return re.sub(r"/v1/?$", "", self.s.base_url)

    def _cache_path(self, payload: dict) -> Path:
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]
        return self._cache_dir / f"{h}.json"

    def health(self) -> dict:
        """Cheap connectivity probe used by the CLI before a long run."""
        try:
            if self.backend == "ollama":
                r = requests.get(f"{self._ollama_root}/api/tags", timeout=5)
                r.raise_for_status()
                names = [m["name"] for m in r.json().get("models", [])]
                ok = any(n == self.s.model or n.split(":")[0] == self.s.model for n in names) or self.s.model in names
                return {"reachable": True, "model_available": ok, "models": names}
            r = requests.get(f"{self.s.base_url}/models", headers=self._headers(), timeout=5)
            r.raise_for_status()
            return {"reachable": True, "model_available": True, "models": [m.get("id") for m in r.json().get("data", [])]}
        except Exception as e:  # noqa: BLE001
            return {"reachable": False, "model_available": False, "error": str(e)}

    def _alive(self) -> bool:
        return bool(self.health().get("reachable"))

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.s.api_key}", "Content-Type": "application/json"}

    def chat(
        self,
        messages: list[dict],
        *,
        stage: str = "misc",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        json_schema: Optional[dict] = None,
        json_mode: bool = False,
        max_tokens: int = 1500,
        retry: bool = False,
    ) -> str:
        """One raw call. Returns the assistant text. Cached; uncached calls are budget-checked.

        Transport policy (fail fast instead of hanging for hours on a dead server):
        - connection refused -> wait briefly, re-probe; still down -> BackendUnavailable
        - read timeout -> retry once if the server still answers a cheap probe; a second
          timeout means the model is stalled (typically RAM thrash) -> BackendUnavailable
        - HTTP 5xx mentioning memory -> BackendUnavailable (Ollama OOM)
        """
        model = model or self.s.model
        temperature = self.s.temperature if temperature is None else temperature
        key = {"m": model, "msgs": messages, "t": temperature, "schema": json_schema, "json": json_mode, "b": self.backend}
        cpath = self._cache_path(key)
        if self.use_cache and cpath.exists():
            try:  # cached replies cost nothing, so they are served even when the budget is spent (cheap resume)
                return json.loads(cpath.read_text(encoding="utf-8"))["text"]
            except Exception:  # noqa: BLE001
                pass
        self.budget.check()

        t0 = time.monotonic()
        last_exc: Optional[Exception] = None
        timeouts = 0
        for net_attempt in range(3):
            try:
                text, pt, ct = self._call(messages, model, temperature, json_schema, json_mode, max_tokens)
                break
            except requests.HTTPError as e:
                body = (e.response.text if e.response is not None else "")[:300]
                if re.search(r"memory|out of memory|oom|insufficient", body, re.I):
                    self.budget.record_failure(stage)
                    raise BackendUnavailable(f"[{stage}] model server ran out of memory: {body.strip()}") from e
                # Grammar-constrained schema rejected by server -> degrade to plain JSON mode once.
                if json_schema is not None and e.response is not None and e.response.status_code in (400, 422, 500):
                    json_schema, json_mode = None, True
                last_exc = e
            except requests.Timeout as e:
                last_exc, timeouts = e, timeouts + 1
                if timeouts >= 2 or not self._alive():
                    self.budget.record_failure(stage)
                    raise BackendUnavailable(
                        f"[{stage}] model server stalled: {timeouts} request(s) exceeded {self.s.request_timeout}s"
                        + ("" if self._alive() else " and the server no longer responds")) from e
            except requests.ConnectionError as e:
                last_exc = e
                time.sleep(3 * (net_attempt + 1))
                if not self._alive():
                    if net_attempt >= 1:
                        self.budget.record_failure(stage)
                        raise BackendUnavailable(f"[{stage}] model server at {self.s.base_url} is not reachable: {e}") from e
        else:
            self.budget.record_failure(stage)
            raise LLMError(f"[{stage}] LLM request failed after retries: {last_exc}")

        dt = time.monotonic() - t0
        self.budget.record(stage, pt, ct, dt, retry=retry)
        if self.use_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text

    def _call(self, messages, model, temperature, json_schema, json_mode, max_tokens) -> tuple[str, int, int]:
        if self.backend == "ollama":
            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens},
            }
            if json_schema is not None:
                body["format"] = json_schema
            elif json_mode:
                body["format"] = "json"
            r = requests.post(f"{self._ollama_root}/api/chat", json=body, timeout=self.s.request_timeout)
            r.raise_for_status()
            d = r.json()
            return d["message"]["content"], int(d.get("prompt_eval_count", 0)), int(d.get("eval_count", 0))

        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_schema is not None or json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.s.base_url}/chat/completions", json=body, headers=self._headers(), timeout=self.s.request_timeout)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return d["choices"][0]["message"]["content"] or "", int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))

    # ---------------------------------------------------------------- structured
    def structured(
        self,
        schema: Type[T],
        system: str,
        user: str,
        *,
        stage: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1500,
        max_retries: Optional[int] = None,
    ) -> T:
        """Schema-enforced call with a self-healing retry loop."""
        retries = self.s.max_retries if max_retries is None else max_retries
        sys_msg = (
            f"{system.strip()}\n\nRespond with ONE JSON object only - no prose, no markdown fences. "
            f"It must match this shape:\n{json.dumps(skeleton(schema), indent=1)}"
        )
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        json_schema = schema.model_json_schema() if self.backend == "ollama" else None
        raw, errors = "", ""
        for attempt in range(retries + 1):
            raw = self.chat(
                messages, stage=stage, model=model, json_schema=json_schema, json_mode=True,
                temperature=temperature if attempt == 0 else max(0.3, temperature or self.s.temperature),
                max_tokens=max_tokens, retry=attempt > 0,
            )
            try:
                data = _coerce_top_level(extract_json(raw), schema)
                return schema.model_validate(data)
            except ValueError as e:  # includes ValidationError
                errors = _format_errors(e) if isinstance(e, ValidationError) else str(e)
            messages = messages + [
                {"role": "assistant", "content": raw[:3000]},
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply was rejected by the validator:\n{errors}\n"
                        "Return the corrected JSON object only, with every required field present."
                    ),
                },
            ]
        self.budget.record_failure(stage)
        raise StructuredOutputError(f"[{stage}] invalid structured output after {retries + 1} attempts", raw, errors)

    # ---------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], batch: int = 32) -> Optional[list[list[float]]]:
        """Return embeddings, or None if no embedding model is reachable (caller falls back)."""
        if not self.s.embed_model or self._embed_ok is False or not texts:
            return None
        out: list[list[float]] = []
        try:
            for i in range(0, len(texts), batch):
                chunk = [t[:4000] for t in texts[i : i + batch]]
                t0 = time.monotonic()
                if self.backend == "ollama":
                    r = requests.post(
                        f"{self._ollama_root}/api/embed",
                        json={"model": self.s.embed_model, "input": chunk},
                        timeout=self.s.request_timeout,
                    )
                    r.raise_for_status()
                    vecs = r.json()["embeddings"]
                else:
                    r = requests.post(
                        f"{self.s.base_url}/embeddings",
                        json={"model": self.s.embed_model, "input": chunk},
                        headers=self._headers(),
                        timeout=self.s.request_timeout,
                    )
                    r.raise_for_status()
                    vecs = [d["embedding"] for d in r.json()["data"]]
                self.budget.record("embed", sum(len(c) // 4 for c in chunk), 0, time.monotonic() - t0)
                out.extend(vecs)
            self._embed_ok = True
            return out
        except Exception:  # noqa: BLE001
            self._embed_ok = False
            return None


def load_prompt(name: str, settings: Optional[Settings] = None, **vars: Any) -> str:
    """Load prompts/<name>.md and substitute {{var}} placeholders (braces in JSON stay intact)."""
    s = settings or get_settings()
    text = (s.prompts_dir / f"{name}.md").read_text(encoding="utf-8")
    missing = set(re.findall(r"\{\{(\w+)\}\}", text)) - set(vars)
    if missing:
        raise KeyError(f"prompt {name!r} has unfilled placeholders: {sorted(missing)}")
    vals = {k: (v if isinstance(v, str) else json.dumps(v, indent=1, ensure_ascii=False, default=str)) for k, v in vars.items()}
    # single pass, so placeholder-like text inside substituted documents is never re-expanded
    return re.sub(r"\{\{(\w+)\}\}", lambda m: vals.get(m.group(1), m.group(0)), text)


def split_prompt(text: str) -> tuple[str, str]:
    """Prompt files hold '## SYSTEM' and '## USER' sections."""
    m = re.search(r"^##\s*SYSTEM\s*$(.*?)^##\s*USER\s*$(.*)", text, re.S | re.M)
    if not m:
        return "You are a precise assistant that outputs JSON.", text
    return m.group(1).strip(), m.group(2).strip()
