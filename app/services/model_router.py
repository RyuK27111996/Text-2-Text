from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal


RoutingProfile = Literal["auto", "text_fast", "text_quality", "cheap_text", "vision_text"]


@dataclass(slots=True)
class ModelHealth:
    """Track transient health and cooldown state for one routed model."""

    in_flight: int = 0
    consecutive_rate_limits: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None


@dataclass(slots=True)
class ModelRouter:
    """Route requests across ordered Gemini fallback pools."""

    model_pools: dict[RoutingProfile, list[str]]
    model_aliases: dict[str, str] = field(default_factory=dict)
    _health: dict[str, ModelHealth] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        """Initialize health entries for every known model."""
        for models in self.model_pools.values():
            for model in models:
                self._health.setdefault(model, ModelHealth())

    def resolve_profile(self, explicit_model: str | None, wants_json: bool) -> RoutingProfile:
        """Choose a routing profile for the incoming request."""
        if explicit_model:
            normalized = self.model_aliases.get(explicit_model, explicit_model)
            for profile, models in self.model_pools.items():
                if profile != "auto" and normalized in models:
                    return profile
        if wants_json:
            return "text_fast"
        return "text_fast"

    def normalize_model_name(self, model: str) -> str:
        """Map known aliases or common mistakes to routed model names."""
        return self.model_aliases.get(model, model)

    async def get_candidates(self, profile: RoutingProfile, preferred_model: str | None = None) -> list[str]:
        """Return healthy models in preferred order for a routing profile."""
        async with self._lock:
            now = time.monotonic()
            candidates = list(self.model_pools[profile])
            if preferred_model:
                preferred_model = self.normalize_model_name(preferred_model)
                if preferred_model in candidates:
                    candidates.remove(preferred_model)
                    candidates.insert(0, preferred_model)

            healthy: list[tuple[int, str]] = []
            cooling: list[tuple[int, str]] = []
            for order, model in enumerate(candidates):
                health = self._health.setdefault(model, ModelHealth())
                if health.cooldown_until > now:
                    cooling.append((order, model))
                else:
                    healthy.append((order, model))

            # Prefer healthy models first, then cooling models as a last resort.
            return [model for _, model in healthy] + [model for _, model in cooling]

    async def mark_in_flight(self, model: str, delta: int) -> None:
        """Track the number of active requests using a model."""
        async with self._lock:
            health = self._health.setdefault(model, ModelHealth())
            health.in_flight = max(0, health.in_flight + delta)

    async def mark_success(self, model: str) -> None:
        """Reset rate-limit penalties after a successful model call."""
        async with self._lock:
            health = self._health.setdefault(model, ModelHealth())
            health.consecutive_rate_limits = 0
            health.cooldown_until = 0.0
            health.last_error = None

    async def mark_rate_limited(self, model: str, detail: str | None = None) -> None:
        """Move a rate-limited model into cooldown before it is retried."""
        async with self._lock:
            health = self._health.setdefault(model, ModelHealth())
            health.consecutive_rate_limits += 1
            cooldown_seconds = self._cooldown_seconds(health.consecutive_rate_limits)
            health.cooldown_until = time.monotonic() + cooldown_seconds
            health.last_error = detail

    async def mark_failure(self, model: str, detail: str | None = None) -> None:
        """Record a non-rate-limit failure without cooling down the model."""
        async with self._lock:
            health = self._health.setdefault(model, ModelHealth())
            health.last_error = detail

    @staticmethod
    def _cooldown_seconds(consecutive_rate_limits: int) -> int:
        """Return the cooldown duration for consecutive 429 responses."""
        if consecutive_rate_limits <= 1:
            return 15
        if consecutive_rate_limits == 2:
            return 60
        return 300


def build_default_model_router() -> ModelRouter:
    """Create the default Gemma routing policy used by the application."""
    pools: dict[RoutingProfile, list[str]] = {
        "auto": ["gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it"],
        "text_fast": [
            "gemma-3-27b-it",
            "gemma-3-12b-it",
            "gemma-3-4b-it",
            "gemma-3-2b-it",
            "gemma-3-1b-it",
        ],
        "text_quality": [
            "gemma-3-27b-it",
            "gemma-3-12b-it",
            "gemma-3-4b-it",
        ],
        "cheap_text": [
            "gemma-3-4b-it",
            "gemma-3-2b-it",
            "gemma-3-1b-it",
        ],
        "vision_text": [
            "gemma-3-27b-it",
            "gemma-3-12b-it",
            "gemma-3-4b-it",
        ],
    }
    aliases = {
        "gemma-3-27b": "gemma-3-27b-it",
        "gemma-3-12b": "gemma-3-12b-it",
        "gemma-3-4b": "gemma-3-4b-it",
        "gemma-3-2b": "gemma-3-2b-it",
        "gemma-3-1b": "gemma-3-1b-it",
    }
    return ModelRouter(model_pools=pools, model_aliases=aliases)
