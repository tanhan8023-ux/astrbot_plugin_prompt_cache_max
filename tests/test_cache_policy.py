from pathlib import Path

from astrbot_plugin_prompt_cache_max.cache_policy import (
    LightState,
    PrefixInfo,
    DEFAULT_STABLE_CACHE_ANCHOR_TEXT,
    STABLE_STYLE_START,
    analyze_prefix_risks_from_payload,
    apply_stable_style_rules_to_payload,
    base_url_is_allowlisted,
    inject_payload,
    merge_config,
    normalize_provider_with_config,
    openai_retention_allowed,
    with_stable_style_rules,
    estimate_tokens,
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
    config = merge_config({"cache_injection_enabled": True})
    state = make_state(tmp_path)
    allowed = inject_payload({}, make_info("openai", True), config, state)
    denied = inject_payload({}, make_info("openai", False), config, state)
    assert allowed.injected is True
    assert allowed.payload["prompt_cache_key"] == "b" * 64
    assert "extra_body" not in allowed.payload
    assert "custom_extra_body" not in allowed.payload
    assert "prompt_cache_retention" not in allowed.payload
    assert denied.injected is False
    assert "prompt_cache_key" not in denied.payload


def test_openai_stream_payload_requests_usage(tmp_path: Path):
    config = merge_config({"cache_injection_enabled": True})
    state = make_state(tmp_path)
    result = inject_payload({"stream": True}, make_info("openai", True), config, state)
    assert result.injected is True
    assert result.payload["stream_options"]["include_usage"] is True


def test_openai_cache_key_extra_body_can_be_disabled(tmp_path: Path):
    config = merge_config({"cache_injection_enabled": True, "openai_cache_key_extra_body": False})
    state = make_state(tmp_path)
    result = inject_payload({}, make_info("openai", True), config, state)
    assert result.injected is True
    assert result.payload["prompt_cache_key"] == "b" * 64
    assert "extra_body" not in result.payload
    assert "custom_extra_body" not in result.payload


def test_cache_injection_disabled_by_default(tmp_path: Path):
    config = merge_config({})
    state = make_state(tmp_path)
    result = inject_payload({}, make_info("openai", True), config, state)
    assert result.injected is False
    assert result.note == "cache injection disabled"
    assert "prompt_cache_key" not in result.payload


def test_aiwork_is_allowlisted_and_openai_compatible():
    config = merge_config({})
    assert base_url_is_allowlisted("https://aiwork.fans/v1", list(config.get("allowlist_base_urls", []))) is True
    assert normalize_provider_with_config("anthropic", "claude-opus-4-6", "https://aiwork.fans/v1", config) == "openai"


def test_aiwork_openai_payload_only_sends_cache_key(tmp_path: Path):
    config = merge_config({"cache_injection_enabled": True})
    state = make_state(tmp_path)
    info = PrefixInfo(
        provider="openai",
        model="claude-opus-4-6",
        base_url="https://aiwork.fans/v1",
        base_url_host="aiwork.fans",
        fingerprint="a" * 64,
        cache_key_fingerprint="b" * 64,
        token_estimate=5000,
        allowlisted=True,
    )
    result = inject_payload({}, info, config, state)
    assert result.injected is True
    assert result.payload["prompt_cache_key"] == "b" * 64
    assert "extra_body" not in result.payload
    assert "prompt_cache_retention" not in result.payload


def test_aiwork_never_sends_retention_even_when_enabled(tmp_path: Path):
    config = merge_config(
        {
            "cache_injection_enabled": True,
            "openai_prompt_cache_retention": {"enabled": True, "value": "24h"},
        }
    )
    state = make_state(tmp_path)
    info = PrefixInfo(
        provider="openai",
        model="gemini-3-flash-preview",
        base_url="https://aiwork.fans/v1",
        base_url_host="aiwork.fans",
        fingerprint="a" * 64,
        cache_key_fingerprint="b" * 64,
        token_estimate=5000,
        allowlisted=True,
    )
    result = inject_payload({}, info, config, state)
    assert result.injected is True
    assert result.payload["prompt_cache_key"] == "b" * 64
    assert "prompt_cache_retention" not in result.payload
    assert openai_retention_allowed(info, config) is False


def test_openai_near_threshold_still_injects(tmp_path: Path):
    config = merge_config({"cache_injection_enabled": True, "min_prefix_tokens": {"openai": 1024}})
    state = make_state(tmp_path)
    result = inject_payload({}, make_info("openai", True, tokens=1019), config, state)
    assert result.injected is True
    assert result.note == "prompt_cache_key:near_threshold"
    assert result.payload["prompt_cache_key"] == "b" * 64


def test_openai_far_below_threshold_does_not_inject(tmp_path: Path):
    config = merge_config({"cache_injection_enabled": True, "min_prefix_tokens": {"openai": 1024}})
    state = make_state(tmp_path)
    result = inject_payload({}, make_info("openai", True, tokens=900), config, state)
    assert result.injected is False
    assert result.note == "prefix below threshold"


def test_claude_cache_control_limit_and_ttl(tmp_path: Path):
    config = merge_config({"cache_injection_enabled": True, "max_claude_cache_blocks": 2, "cache_ttl": {"anthropic": "5m"}})
    state = make_state(tmp_path)
    payload = {"system": "stable rules", "tools": [{"name": "a"}, {"name": "b"}]}
    result = inject_payload(payload, make_info("anthropic", True), config, state)
    assert result.injected is True
    blocks = result.payload["system"] + result.payload["tools"]
    tagged = [block for block in blocks if "cache_control" in block]
    assert len(tagged) == 2
    assert tagged[0]["cache_control"]["ttl"] == "5m"


def test_gemini_reuses_cache_until_expired(tmp_path: Path):
    config = merge_config({"cache_injection_enabled": True, "cache_ttl": {"gemini": "3600s"}})
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
    state.get_or_create_gemini_cache("b" * 64, "gemini", "gemini-pro", 5000, 3600, lambda *_args: "cached/1")
    text = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "system prompt" not in text
    assert "user message" not in text
    assert "gemini-pro" in text


def test_prefix_history_tracks_same_fingerprint_without_prompt_text(tmp_path: Path):
    state = make_state(tmp_path)
    info = make_info("openai", True)
    first_risk = analyze_prefix_risks_from_payload({"messages": [{"role": "system", "content": "固定人设"}]})
    state.remember_inspect(info, True, "prompt_cache_key", first_risk)
    assert state.last_inspect["prefix_same_as_previous"] is None

    state.remember_inspect(info, True, "prompt_cache_key", first_risk)
    assert state.last_inspect["prefix_same_as_previous"] is True

    text = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "固定人设" not in text


def test_prefix_risk_detects_dynamic_content_near_front():
    report = analyze_prefix_risks_from_payload(
        {
            "messages": [
                {"role": "system", "content": "Current datetime: 2026-05-29 00:06\n状态栏: 在线"},
                {"role": "user", "content": "你好"},
            ]
        }
    )
    assert report.dynamic_prefix is True
    assert "dynamic_content_near_front" in report.reasons


def test_prefix_risk_ignores_dynamic_content_after_cache_window():
    report = analyze_prefix_risks_from_payload(
        {
            "messages": [
                {"role": "system", "content": "固定人设" * 900 + "\n当前时间: 2026-05-29 00:06"},
                {"role": "user", "content": "你好"},
            ]
        }
    )
    assert report.first_dynamic_token_estimate is not None
    assert report.first_dynamic_token_estimate >= 1024
    assert report.dynamic_prefix is False
    assert "dynamic_content_near_front" not in report.reasons


def test_prefix_risk_detects_media_near_front():
    report = analyze_prefix_risks_from_payload(
        {
            "messages": [
                {"role": "system", "content": "固定人设"},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/gif;base64,xxx"}}]},
            ]
        }
    )
    assert report.media_prefix is True
    assert "media_near_front" in report.reasons


def test_stable_style_rules_prepend_once():
    config = merge_config({"stable_style_rules": {"enabled": True}})
    updated, inserted = with_stable_style_rules("原本人设", config)
    assert inserted is True
    assert updated.startswith(STABLE_STYLE_START)
    assert "缓存稳定锚点" in updated
    assert updated.endswith("原本人设")

    updated_again, inserted_again = with_stable_style_rules(updated, config)
    assert inserted_again is False
    assert updated_again == updated


def test_stable_style_rules_can_be_disabled():
    config = merge_config({"stable_style_rules": {"enabled": False}})
    updated, inserted = with_stable_style_rules("原本人设", config)
    assert inserted is False
    assert updated == "原本人设"


def test_stable_cache_anchor_can_be_disabled():
    config = merge_config({"stable_style_rules": {"enabled": True, "cache_anchor_enabled": False}})
    updated, inserted = with_stable_style_rules("原本人设", config)
    assert inserted is True
    assert "缓存稳定锚点" not in updated
    assert DEFAULT_STABLE_CACHE_ANCHOR_TEXT not in updated


def test_stable_cache_anchor_pads_to_target_tokens():
    config = merge_config({"stable_style_rules": {"enabled": True, "cache_anchor_target_tokens": 1536}})
    updated, inserted = with_stable_style_rules("原本人设", config)
    assert inserted is True
    assert estimate_tokens(updated) >= 1536


def test_cache_anchor_auto_when_cache_injection_enabled():
    config = merge_config({"cache_injection_enabled": True})
    updated, inserted = with_stable_style_rules("原本人设", config)
    assert inserted is True
    assert "缓存稳定锚点" in updated
    assert estimate_tokens(updated) >= 3072


def test_stable_style_rules_apply_to_payload_messages():
    config = merge_config({"stable_style_rules": {"enabled": True}})
    payload = {"messages": [{"role": "user", "content": "你好"}]}
    assert apply_stable_style_rules_to_payload(payload, config) is True
    assert payload["messages"][0]["role"] == "system"
    assert STABLE_STYLE_START in payload["messages"][0]["content"]


def test_actual_prefix_history_tracks_real_payload(tmp_path: Path):
    state = make_state(tmp_path)
    info = make_info("openai", True)
    first_risk = analyze_prefix_risks_from_payload(
        {"messages": [{"role": "system", "content": "固定人设"}, {"role": "user", "content": "a"}]}
    )
    second_risk = analyze_prefix_risks_from_payload(
        {"messages": [{"role": "system", "content": "固定人设"}, {"role": "user", "content": "b"}]}
    )
    state.remember_inspect(info, True, "prompt_cache_key", first_risk)
    state.remember_inspect(info, True, "prompt_cache_key", second_risk)
    assert state.last_inspect["prefix_same_as_previous"] is True
    assert state.last_inspect["actual_prefix_same_as_previous"] is False
