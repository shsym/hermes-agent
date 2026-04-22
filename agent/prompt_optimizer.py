"""Optimizer hook for the AIAgent system-prompt assembly.

Hermes-agent provides an IdentityOptimizer (no-op) by default. Downstream
forks (pie-hermes) register concrete optimizers that reshape the list of
prompt parts before the final join — for example, dropping the
model-family guidance block when the inferlet can inject the equivalent KV
segment out-of-band."""
from __future__ import annotations
from typing import Protocol


class PromptOptimizer(Protocol):
    def optimize(self, parts: list[str]) -> list[str]: ...


class IdentityOptimizer:
    """Default no-op. Returns parts unchanged."""
    def optimize(self, parts: list[str]) -> list[str]:
        return parts


_registered: PromptOptimizer = IdentityOptimizer()


def register(opt: PromptOptimizer) -> None:
    """Install a new optimizer. Idempotent; call reset_default() to restore."""
    global _registered
    _registered = opt


def reset_default() -> None:
    global _registered
    _registered = IdentityOptimizer()


def get() -> PromptOptimizer:
    return _registered
