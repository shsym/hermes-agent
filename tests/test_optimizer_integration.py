"""Integration: register a PromptOptimizer that emits a header, call
_build_system_prompt with a minimal AIAgent, and verify the header
survives _merge_optimizer_headers onto a chat.completions.create kwargs dict.

Skipped if the host's test venv lacks AIAgent's runtime deps (openai etc.).
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_optimizer():
    """Always start + end each test with the identity optimizer."""
    try:
        from agent.prompt_optimizer import reset_default
    except ImportError:
        pytest.skip("hermes-agent prompt_optimizer not importable")
    reset_default()
    yield
    reset_default()


def test_optimizer_header_flows_from_build_to_kwargs():
    """End-to-end: optimizer emits a header → build_system_prompt stashes it
    on self._prompt_optimizer_extra_headers → _merge_optimizer_headers adds
    it to a chat.completions.create kwargs dict."""

    # Guard: AIAgent needs several runtime deps; skip cleanly if absent.
    try:
        from run_agent import AIAgent  # noqa: F401
    except Exception as e:
        pytest.skip(f"AIAgent not importable in this venv: {type(e).__name__}: {e}")

    from agent.prompt_optimizer import PromptOptimization, register

    class StubOptimizer:
        def optimize(self, parts):
            return PromptOptimization(
                parts=parts, extra_headers={"X-Hermes-Variant": "codex"}
            )

    register(StubOptimizer())

    # Build a minimal AIAgent-like context without calling __init__, which
    # requires many runtime params (OpenAI client, config, memory_store…).
    # We only need the attributes _build_system_prompt and
    # _merge_optimizer_headers read.
    agent = AIAgent.__new__(AIAgent)
    agent.valid_tool_names = set()
    agent.skip_context_files = True
    agent.model = "stub-gpt-for-integration"
    agent.provider = "custom"
    agent.platform = ""
    agent.ephemeral_system_prompt = None
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_manager = None
    agent._tool_use_enforcement = True
    agent.pass_session_id = False
    agent.session_id = None
    agent._prompt_optimizer_extra_headers = {}

    # _build_system_prompt does many things; on a partial agent instance it
    # may fail on attributes we haven't populated. To stay honest without
    # full fidelity, call the method and tolerate failure — we only care
    # that the optimizer hook side-effect sets _prompt_optimizer_extra_headers.
    try:
        _ = agent._build_system_prompt(system_message=None)
    except Exception:
        # Partial fixture couldn't complete; but the optimizer hook runs
        # at the tail of _build_system_prompt. If it didn't run at all,
        # the attribute stays {} and the assertion below catches it.
        pass

    assert agent._prompt_optimizer_extra_headers.get("X-Hermes-Variant") == "codex", (
        "Optimizer's extra_headers did not flow onto self._prompt_optimizer_extra_headers. "
        f"Got: {agent._prompt_optimizer_extra_headers!r}"
    )

    # Now assert _merge_optimizer_headers correctly applies them.
    kwargs = {"model": agent.model, "messages": [{"role": "user", "content": "hi"}]}
    agent._merge_optimizer_headers(kwargs)
    assert "extra_headers" in kwargs
    assert kwargs["extra_headers"].get("X-Hermes-Variant") == "codex"


def test_merge_preserves_caller_headers():
    """_merge_optimizer_headers must not clobber caller-supplied headers on conflict."""
    try:
        from run_agent import AIAgent  # noqa: F401
    except Exception as e:
        pytest.skip(f"AIAgent not importable: {e}")

    agent = AIAgent.__new__(AIAgent)
    agent._prompt_optimizer_extra_headers = {"X-Hermes-Variant": "codex"}

    # Caller-supplied headers should win when both name the same key.
    kwargs = {"extra_headers": {"X-Hermes-Variant": "override-from-caller"}}
    agent._merge_optimizer_headers(kwargs)
    assert kwargs["extra_headers"]["X-Hermes-Variant"] == "override-from-caller"
