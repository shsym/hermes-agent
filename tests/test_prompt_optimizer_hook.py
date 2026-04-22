"""PromptOptimizer hook: register/get/reset lifecycle + default behavior."""
from agent.prompt_optimizer import (
    IdentityOptimizer, PromptOptimization, PromptOptimizer,
    _coerce, get, register, reset_default,
)


def test_default_optimizer_is_identity():
    reset_default()
    parts = ["alpha", "beta", "gamma"]
    out = _coerce(get().optimize(parts))
    assert out.parts == parts


def test_register_installs_custom_optimizer():
    calls = []

    class Recorder:
        def optimize(self, parts: list[str]) -> list[str]:
            calls.append(list(parts))
            return parts

    reset_default()
    register(Recorder())
    try:
        out = _coerce(get().optimize(["x", "y"]))
    finally:
        reset_default()
    assert calls == [["x", "y"]]
    assert out.parts == ["x", "y"]


def test_reset_restores_identity():
    class Dropper:
        def optimize(self, parts: list[str]) -> list[str]:
            return []

    register(Dropper())
    assert _coerce(get().optimize(["x"])).parts == []
    reset_default()
    assert _coerce(get().optimize(["x"])).parts == ["x"]


def test_optimizer_can_drop_parts():
    """Real usage: optimizer removes a section (simulates Idea A)."""
    class DropSecond:
        def optimize(self, parts: list[str]) -> list[str]:
            return [p for i, p in enumerate(parts) if i != 1]

    reset_default()
    register(DropSecond())
    try:
        out = _coerce(get().optimize(["a", "b", "c"]))
    finally:
        reset_default()
    assert out.parts == ["a", "c"]


def test_default_optimizer_returns_empty_headers():
    """IdentityOptimizer returns parts unchanged AND no headers."""
    reset_default()
    result = get().optimize(["a", "b"])
    assert result.parts == ["a", "b"]
    assert result.extra_headers == {}


def test_optimizer_can_emit_extra_headers():
    from agent.prompt_optimizer import PromptOptimization

    class WithHeaders:
        def optimize(self, parts):
            return PromptOptimization(parts=parts, extra_headers={"X-Foo": "bar"})

    reset_default()
    register(WithHeaders())
    try:
        result = get().optimize(["x"])
    finally:
        reset_default()
    assert result.extra_headers == {"X-Foo": "bar"}
