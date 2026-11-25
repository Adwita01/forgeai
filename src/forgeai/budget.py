"""Token / latency accounting with a hard budget and per-stage breakdown."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class StageUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    retries: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Budget:
    max_tokens: int = 0  # 0 = unlimited
    max_seconds: float = 0.0  # wall clock, 0 = unlimited
    started: float = field(default_factory=time.monotonic)
    stages: dict[str, StageUsage] = field(default_factory=lambda: defaultdict(StageUsage))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, stage: str, prompt_tokens: int, completion_tokens: int, seconds: float, retry: bool = False) -> None:
        with self._lock:
            s = self.stages[stage]
            s.calls += 1
            s.prompt_tokens += prompt_tokens
            s.completion_tokens += completion_tokens
            s.seconds += seconds
            if retry:
                s.retries += 1

    def record_failure(self, stage: str) -> None:
        with self._lock:
            self.stages[stage].failures += 1

    @property
    def tokens_used(self) -> int:
        return sum(s.total_tokens for s in self.stages.values())

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining_fraction(self) -> float:
        fr = [1.0]
        if self.max_tokens:
            fr.append(1 - self.tokens_used / self.max_tokens)
        if self.max_seconds:
            fr.append(1 - self.elapsed / self.max_seconds)
        return max(0.0, min(fr))

    def exhausted(self) -> bool:
        return self.remaining_fraction() <= 0.0

    def check(self) -> None:
        if self.exhausted():
            raise BudgetExceeded(
                f"budget exhausted: {self.tokens_used} tokens / {self.elapsed:.0f}s "
                f"(limits {self.max_tokens or 'inf'} tokens / {self.max_seconds or 'inf'}s)"
            )

    def summary(self) -> dict:
        return {
            "tokens_used": self.tokens_used,
            "elapsed_s": round(self.elapsed, 1),
            "max_tokens": self.max_tokens,
            "max_seconds": self.max_seconds,
            "stages": {
                k: {
                    "calls": v.calls,
                    "prompt_tokens": v.prompt_tokens,
                    "completion_tokens": v.completion_tokens,
                    "seconds": round(v.seconds, 1),
                    "schema_retries": v.retries,
                    "hard_failures": v.failures,
                }
                for k, v in sorted(self.stages.items())
            },
        }
