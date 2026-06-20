from astrbot_plugin_prompt_cache_max.main import PromptCacheMaxPlugin


def test_verdict_prefers_stable_prefix_over_risk_warning(tmp_path):
    class Context:
        data_dir = tmp_path

    plugin = PromptCacheMaxPlugin(
        Context(),
        {
            "provider_wrapping_enabled": True,
            "cache_injection_enabled": True,
        },
    )
    info = {
        "provider": "openai",
        "model": "gemini-3-flash-preview",
        "base_url_host": "aiwork.fans",
        "allowlisted": True,
        "token_estimate": 1020,
        "injected": True,
        "prefix_same_as_previous": True,
        "usage_observed": True,
        "observed_cached_tokens": 0,
        "dynamic_prefix": True,
    }
    assert plugin._format_cache_verdict(info) == "部分稳定：稳定前缀一致，下一轮继续看真实请求前缀"


def test_inject_session_header_creates_extra_headers(tmp_path):
    class Context:
        data_dir = tmp_path

    class Result:
        session_cache_enabled = True
        session_header_name = "session_id"
        session_header_value = "stable-session"

    plugin = PromptCacheMaxPlugin(Context(), {})
    kwargs = {}
    assert plugin._inject_session_header("openai", kwargs, Result()) is True
    assert kwargs["extra_headers"]["session_id"] == "stable-session"


def test_inject_session_header_preserves_existing_headers(tmp_path):
    class Context:
        data_dir = tmp_path

    class Result:
        session_cache_enabled = True
        session_header_name = "session_id"
        session_header_value = "stable-session"

    plugin = PromptCacheMaxPlugin(Context(), {})
    kwargs = {"extra_headers": {"X-Test": "1"}}
    assert plugin._inject_session_header("openai", kwargs, Result()) is True
    assert kwargs["extra_headers"]["X-Test"] == "1"
    assert kwargs["extra_headers"]["session_id"] == "stable-session"
