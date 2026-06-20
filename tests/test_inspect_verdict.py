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
    assert plugin._format_cache_verdict(info) == "可能未透传：前缀一致但 cached_tokens 为 0，上游可能不支持或没返回统计"
