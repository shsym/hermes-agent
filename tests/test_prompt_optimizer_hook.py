"""PromptOptimizer hook: register/get/reset lifecycle + default behavior."""
from agent.prompt_optimizer import (
    IdentityOptimizer, PromptOptimizer, get, register, reset_default,
)


def test_default_optimizer_is_identity():
    reset_default()
    parts = ["alpha", "beta", "gamma"]
    out = get().optimize(parts)
    assert out == parts


def test_register_installs_custom_optimizer():
    calls = []

    class Recorder:
        def optimize(self, parts: list[str]) -> list[str]:
            calls.append(list(parts))
            return parts

    reset_default()
    register(Recorder())
    try:
        out = get().optimize(["x", "y"])
    finally:
        reset_default()
    assert calls == [["x", "y"]]
    assert out == ["x", "y"]


def test_reset_restores_identity():
    class Dropper:
        def optimize(self, parts: list[str]) -> list[str]:
            return []

    register(Dropper())
    assert get().optimize(["x"]) == []
    reset_default()
    assert get().optimize(["x"]) == ["x"]


def test_optimizer_can_drop_parts():
    """Real usage: optimizer removes a section (simulates Idea A)."""
    class DropSecond:
        def optimize(self, parts: list[str]) -> list[str]:
            return [p for i, p in enumerate(parts) if i != 1]

    reset_default()
    register(DropSecond())
    try:
        out = get().optimize(["a", "b", "c"])
    finally:
        reset_default()
    assert out == ["a", "c"]
