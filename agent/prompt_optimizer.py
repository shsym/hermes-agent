"""Optimizer hook for the AIAgent system-prompt assembly.

Hermes-agent provides an IdentityOptimizer (no-op) by default. Downstream
forks (pie-hermes) register concrete optimizers that reshape the list of
prompt parts — and optionally emit per-request extra_headers — before the
final join and API call."""
from __future__ import annotations

import dataclasses as dc
from typing import Protocol, runtime_checkable


@dc.dataclass(frozen=True)
class PromptOptimization:
    """Result of an optimizer pass.

    - ``parts``: the (possibly reshaped) list of prompt sections.
    - ``extra_headers``: per-request HTTP headers the caller should add to
      the chat-completions call via ``extra_headers=``. Ignored when empty.
    """
    parts: list[str]
    extra_headers: dict[str, str] = dc.field(default_factory=dict)


@runtime_checkable
class PromptOptimizer(Protocol):
    def optimize(self, parts: list[str]) -> PromptOptimization: ...


class IdentityOptimizer:
    """Default no-op."""
    def optimize(self, parts: list[str]) -> PromptOptimization:
        return PromptOptimization(parts=parts)


_registered: PromptOptimizer = IdentityOptimizer()


def register(opt: PromptOptimizer) -> None:
    global _registered
    _registered = opt


def reset_default() -> None:
    global _registered
    _registered = IdentityOptimizer()


def get() -> PromptOptimizer:
    return _registered


def _coerce(result) -> PromptOptimization:
    """Wrap a Phase-0-style bare list return as a PromptOptimization.

    Lets older optimizers that return ``list[str]`` keep working while new
    code returns PromptOptimization directly.
    """
    if isinstance(result, PromptOptimization):
        return result
    if isinstance(result, list):
        return PromptOptimization(parts=result)
    raise TypeError(
        f"PromptOptimizer.optimize must return list[str] or PromptOptimization, "
        f"got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# Ephemeral-text router (Phase-1 Idea G)
# ---------------------------------------------------------------------------

@runtime_checkable
class EphemeralRouter(Protocol):
    def route(self, ephemeral_text: str) -> tuple["str | None", dict[str, str]]: ...


class IdentityEphemeralRouter:
    """Default no-op: ephemeral text flows through to the system prompt
    as before; no extra_headers emitted."""
    def route(self, ephemeral_text: str) -> tuple["str | None", dict[str, str]]:
        return (ephemeral_text, {})


_ephemeral_registered: EphemeralRouter = IdentityEphemeralRouter()


def register_ephemeral_router(r: EphemeralRouter) -> None:
    global _ephemeral_registered
    _ephemeral_registered = r


def reset_ephemeral_router() -> None:
    global _ephemeral_registered
    _ephemeral_registered = IdentityEphemeralRouter()


def get_ephemeral_router() -> EphemeralRouter:
    return _ephemeral_registered
