from pathlib import Path

from astrbot_plugin_prompt_cache_max.cache_policy import (
    LightState,
    PrefixInfo,
    STABLE_STYLE_START,
    apply_stable_style_rules_to_payload,
    inject_payload,
    merge_config,
    with_stable_style_rules,
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
        cache_key_fingerprint="b" * 64,
        token_estimate=tokens,
        allowlisted=base_allowlisted,
    )


def test_openai_payload_only_allowlisted(tmp_path: Path):
    config = merge_config({})
    state = make_state(tmp_path)
    allowed = inject_payload({}, make_info("openai", True), config, state)
    denied = inject_payload({}, make_info("openai", False), config, state)
    assert allowed.injected is True
    assert allowed.payload["prompt_cache_key"] == "b" * 64
    assert "prompt_cache_retention" not in allowed.payload
    assert denied.injected is False
    assert "prompt_cache_key" not in denied.payload


def test_openai_near_threshold_still_injects(tmp_path: Path):
    config = merge_config({"min_prefix_tokens": {"openai": 1024}})
    state = make_state(tmp_path)
    result = inject_payload({}, make_info("openai", True, tokens=1019), config, state)
    assert result.injected is True
    assert result.note == "prompt_cache_key:near_threshold"
    assert result.payload["prompt_cache_key"] == "b" * 64


def test_openai_far_below_threshold_does_not_inject(tmp_path: Path):
    config = merge_config({"min_prefix_tokens": {"openai": 1024}})
    state = make_state(tmp_path)
    result = inject_payload({}, make_info("openai", True, tokens=900), config, state)
    assert result.injected is False
    assert result.note == "prefix below threshold"


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


def test_stable_style_rules_prepend_once():
    config = merge_config({})
    updated, inserted = with_stable_style_rules("原本人设", config)
    assert inserted is True
    assert updated.startswith(STABLE_STYLE_START)
    assert updated.endswith("原本人设")

    updated_again, inserted_again = with_stable_style_rules(updated, config)
    assert inserted_again is False
    assert updated_again == updated


def test_stable_style_rules_can_be_disabled():
    config = merge_config({"stable_style_rules": {"enabled": False}})
    updated, inserted = with_stable_style_rules("原本人设", config)
    assert inserted is False
    assert updated == "原本人设"


def test_stable_style_rules_apply_to_payload_messages():
    config = merge_config({})
    payload = {"messages": [{"role": "user", "content": "你好"}]}
    assert apply_stable_style_rules_to_payload(payload, config) is True
    assert payload["messages"][0]["role"] == "system"
    assert STABLE_STYLE_START in payload["messages"][0]["content"]
