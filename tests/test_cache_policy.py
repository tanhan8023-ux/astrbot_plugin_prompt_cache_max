from pathlib import Path

from astrbot_plugin_prompt_cache_max.cache_policy import (
    LightState,
    PrefixInfo,
    inject_payload,
    merge_config,
)


def make_state(tmp_path: Path) -> LightState:
    return LightState(tmp_path / "state.json")


def make_info(provider: str, base_allowlisted: bool = True, tokens: int = 5000) -> PrefixInfo:
    return PrefixInfo(
        provider=provider,
        model="test-model",
        base_url="https://api.55api.com/v1" if base_allowlisted else "https://unknown.example/v1",
        base_url_host="api.55api.com" if base_allowlisted else "unknown.example",
        fingerprint="a" * 64,
        token_estimate=tokens,
        allowlisted=base_allowlisted,
    )


def test_openai_payload_only_allowlisted(tmp_path: Path):
    config = merge_config({})
    state = make_state(tmp_path)
    allowed = inject_payload({}, make_info("openai", True), config, state)
    denied = inject_payload({}, make_info("openai", False), config, state)
    assert allowed.injected is True
    assert allowed.payload["prompt_cache_key"] == "a" * 64
    assert denied.injected is False
    assert "prompt_cache_key" not in denied.payload


def test_claude_cache_control_limit_and_ttl(tmp_path: Path):
    config = merge_config({"max_claude_cache_blocks": 2, "cache_ttl": {"anthropic": "5m"}})
    state = make_state(tmp_path)
    payload = {"system": "stable rules", "tools": [{"name": "a"}, {"name": "b"}]}
    result = inject_payload(payload, make_info("anthropic", True), config, state)
    assert result.injected is True
    blocks = result.payload["system"] + result.payload["tools"]
    tagged = [block for block in blocks if "cache_control" in block]
    assert len(tagged) == 2
    assert tagged[0]["cache_control"]["ttl"] == "5m"


def test_gemini_reuses_cache_until_expired(tmp_path: Path):
    config = merge_config({"cache_ttl": {"gemini": "3600s"}})
    state = make_state(tmp_path)
    created = []

    def create_cache(fingerprint: str, model: str, ttl: int):
        created.append((fingerprint, model, ttl))
        return f"cached/{len(created)}"

    first = inject_payload({}, make_info("gemini", True), config, state, create_cache)
    second = inject_payload({}, make_info("gemini", True), config, state, create_cache)
    assert first.payload["cachedContent"] == "cached/1"
    assert second.payload["cachedContent"] == "cached/1"
    assert len(created) == 1


def test_state_does_not_store_prompt_text(tmp_path: Path):
    state = make_state(tmp_path)
    state.get_or_create_gemini_cache("b" * 64, "gemini", "gemini-pro", 5000, 3600)
    text = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "system prompt" not in text
    assert "user message" not in text
    assert "gemini-pro" in text
