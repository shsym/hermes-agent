"""Tests for tools/read_context_tool.py — Phase 2.1 (Idea D)."""

import json

import pytest

from tools import read_context_tool
from tools.read_context_tool import (
    clear_cache,
    get_cached_section_ids,
    read_context,
    set_body_cache,
)
from tools.registry import registry


@pytest.fixture(autouse=True)
def _reset_cache():
    """Module-level _BODY_CACHE leaks across tests; reset before + after."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# read_context handler
# ---------------------------------------------------------------------------


def test_read_context_returns_body_for_known_id():
    set_body_cache({"agents.md#coding-style": "Use 4-space indent.\nNo tabs."})
    out = read_context("agents.md#coding-style")
    parsed = json.loads(out)
    assert parsed == {
        "section_id": "agents.md#coding-style",
        "body": "Use 4-space indent.\nNo tabs.",
    }


def test_read_context_unknown_id_returns_error_with_available_list():
    set_body_cache({
        "agents.md#a": "body-a",
        "agents.md#b": "body-b",
    })
    out = read_context("agents.md#missing")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "agents.md#missing" in parsed["error"]
    assert parsed["available"] == ["agents.md#a", "agents.md#b"]


def test_read_context_rejects_empty_string():
    set_body_cache({"agents.md#x": "y"})
    out = read_context("")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "non-empty" in parsed["error"]


def test_read_context_rejects_whitespace_only_string():
    set_body_cache({"agents.md#x": "y"})
    out = read_context("   \t  ")
    parsed = json.loads(out)
    assert "error" in parsed


def test_read_context_rejects_non_string():
    out = read_context(None)  # type: ignore[arg-type]
    parsed = json.loads(out)
    assert "error" in parsed


def test_read_context_strips_section_id_whitespace():
    set_body_cache({"agents.md#a": "body"})
    out = read_context("  agents.md#a  ")
    parsed = json.loads(out)
    assert parsed["body"] == "body"


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def test_set_body_cache_overwrites_prior_cache():
    set_body_cache({"a#1": "first"})
    assert get_cached_section_ids() == ["a#1"]
    set_body_cache({"b#2": "second", "c#3": "third"})
    assert get_cached_section_ids() == ["b#2", "c#3"]


def test_clear_cache_empties_cache():
    set_body_cache({"a#1": "first", "a#2": "second"})
    assert get_cached_section_ids() == ["a#1", "a#2"]
    clear_cache()
    assert get_cached_section_ids() == []


def test_set_body_cache_with_empty_dict_clears():
    set_body_cache({"a#1": "x"})
    set_body_cache({})
    assert get_cached_section_ids() == []


def test_set_body_cache_copies_input():
    """Mutating the input dict after set_body_cache must not affect the cache."""
    src = {"a#1": "x"}
    set_body_cache(src)
    src["a#2"] = "y"  # mutate caller dict
    assert get_cached_section_ids() == ["a#1"]


# ---------------------------------------------------------------------------
# Registry membership
# ---------------------------------------------------------------------------


def test_read_context_is_registered_under_context_files_toolset():
    entry = registry.get_entry("read_context")
    assert entry is not None
    assert entry.toolset == "context_files"


def test_read_context_schema_has_required_section_id():
    entry = registry.get_entry("read_context")
    assert entry is not None
    schema = entry.schema
    assert schema["name"] == "read_context"
    assert "section_id" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["section_id"]


def test_read_context_dispatch_via_registry():
    set_body_cache({"agents.md#x": "body-x"})
    out = registry.dispatch("read_context", {"section_id": "agents.md#x"})
    parsed = json.loads(out)
    assert parsed["body"] == "body-x"


def test_read_context_check_fn_always_true():
    """No external dependencies — toolset must always be available."""
    available = registry.is_toolset_available("context_files")
    assert available is True
