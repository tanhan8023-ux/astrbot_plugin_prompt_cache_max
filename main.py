from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from .cache_policy import (
    BUILTIN_ALLOWLIST_BASE_URLS,
    LightState,
    analyze_prefix_risks_from_payload,
    analyze_prefix_risks_from_request,
    apply_stable_style_rules_to_payload,
    build_prefix_info,
    inject_payload,
    merge_config,
    normalize_provider,
    normalize_provider_with_config,
    openai_retention_allowed,
    stable_hash,
    with_stable_style_rules,
)
from .response_cache import ExactResponseCache

PLUGIN_VERSION = "0.6.3"

try:
    from astrbot.api.event import AstrMessageEvent, filter
    from astrbot.api.star import Context, Star, register
    from astrbot.api import logger
except Exception:  # pragma: no cover - allows local unit tests without AstrBot.
    AstrMessageEvent = Any
    Context = Any
    logger = None

    class Star:
        def __init__(self, *_args: Any, **_kwargs: Any):
            pass

    class _FallbackFilter:
        def command(self, *_args: Any, **_kwargs: Any):
            def deco(fn):
                return fn

            return deco

        def on_llm_request(self, **_kwargs: Any):
            def deco(fn):
                return fn

            return deco

    filter = _FallbackFilter()

    def register(*_args: Any, **_kwargs: Any):
        def deco(cls):
            return cls

        return deco


def _log_info(message: str) -> None:
    if logger:
        logger.info(message)


def _log_warn(message: str) -> None:
    if logger:
        logger.warning(message)


@register(
    "astrbot_plugin_prompt_cache_max",
    "Codex",
    "帮助 OpenAI 兼容接口、Claude 和 Gemini 更容易复用服务端提示词缓存。",
    PLUGIN_VERSION,
)
class PromptCacheMaxPlugin(Star):
    def __init__(self, context: Context, config: Optional[dict[str, Any]] = None):
        super().__init__(context)
        self.context = context
        self.config = merge_config(config)
        self.state = LightState(self._state_path())
        self._wrapped: list[tuple[Any, str, Any]] = []
        self._last_system_hash_by_key: dict[str, str] = {}
        self._latest_info = None
        self._latest_result = None
        self._latest_write_target = "none"
        self._latest_style_rules = "none"
        self._latest_risk_report = None
        self._latest_response_cache = "none"
        self._force_disable_exact_response_cache()
        self.response_cache = ExactResponseCache(self.config)
        self._response_types: dict[str, Any] = {}
        if self._provider_wrapping_enabled():
            self._wrap_known_providers()

    def _state_path(self) -> Path:
        candidates = [
            getattr(self.context, "data_dir", None),
            getattr(self.context, "plugin_data_dir", None),
            getattr(self.context, "base_dir", None),
        ]
        for candidate in candidates:
            if candidate:
                return Path(candidate) / "prompt_cache_max_state.json"
        return Path(__file__).resolve().parent / "data" / "prompt_cache_max_state.json"

    @filter.on_llm_request(priority=20)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any):
        if not self.config.get("enabled", True):
            return
        if not self.config.get("observe_requests_enabled", False):
            self._latest_style_rules = "passthrough"
            return
        if self._provider_wrapping_enabled():
            self._wrap_known_providers()
        self._apply_stable_style_rules_to_request(req)
        provider, model, base_url = self._infer_request_target(req)
        provider_family = normalize_provider_with_config(provider, model, base_url, self.config)
        key = f"{provider_family}:{model}"
        previous_system_hash = self._last_system_hash_by_key.get(key)
        system_hash = stable_hash(getattr(req, "system_prompt", None))
        info = build_prefix_info(req, provider_family, model, base_url, self.config, previous_system_hash)
        risk_report = analyze_prefix_risks_from_request(req)
        self._latest_risk_report = risk_report
        self._last_system_hash_by_key[key] = system_hash
        self._latest_info = info
        if not self._provider_wrapping_enabled():
            self.state.remember_inspect(info, False, "request observed", risk_report)
        if info.dynamic_system_prompt:
            _log_warn(
                "[PromptCacheMax] system_prompt changed for "
                f"{provider_family}/{model}; put per-turn dynamic text in user content to preserve prompt cache."
            )
        _log_info(
            "[PromptCacheMax] observed "
            f"{provider_family}/{model} prefix={info.fingerprint[:12]} tokens~{info.token_estimate} allowlisted={info.allowlisted}"
        )

    @filter.command("pcache")
    async def pcache(self, event: AstrMessageEvent, subcommand: Optional[str] = None):
        command = (subcommand or "stats").strip().lower()
        if command == "stats":
            yield event.plain_result(self._format_stats())
            return
        if command == "inspect":
            yield event.plain_result(self._format_inspect())
            return
        if command == "clear":
            self.state.clear()
            yield event.plain_result("已清除本地轻量缓存状态和统计，不会删除聊天历史。")
            return
        yield event.plain_result("用法：/pcache stats | inspect | clear")

    async def terminate(self):
        for obj, name, original in reversed(self._wrapped):
            try:
                setattr(obj, name, original)
            except Exception as exc:
                _log_warn(f"[PromptCacheMax] failed to restore provider method {name}: {exc}")
        self._wrapped.clear()
        self.state.save()

    def _format_stats(self) -> str:
        if not self.state.stats:
            return "还没有记录到缓存统计。"
        lines = ["缓存统计："]
        for key, stat in sorted(self.state.stats.items()):
            requests = stat.get("requests", 0)
            cached = stat.get("cached_tokens", 0)
            read = stat.get("cache_read_tokens", 0)
            created = stat.get("cache_creation_tokens", 0)
            exact_hits = stat.get("exact_cache_hits", 0)
            exact_writes = stat.get("exact_cache_writes", 0)
            lines.append(
                f"- {key}: 请求数={requests}, 已缓存 token={cached}, "
                f"读取缓存 token={read}, 写入缓存 token={created}, 精确命中={exact_hits}, 精确写入={exact_writes}"
            )
        return "\n".join(lines)

    def _format_inspect(self) -> str:
        info = self.state.last_inspect
        if not info:
            return "还没有观察到缓存请求。若需要检查，请先开启 observe_requests_enabled 并发起一次对话。"
        return (
            "最近一次缓存请求检查：\n"
            f"- 插件版本：{PLUGIN_VERSION}\n"
            f"- 提供商/模型：{info.get('provider')}/{info.get('model')}\n"
            f"- 接口地址：{info.get('base_url')}\n"
            f"- 接口域名：{info.get('base_url_host')}\n"
            f"- 命中判断：{self._format_cache_verdict(info)}\n"
            f"- 前缀指纹：{info.get('fingerprint')}\n"
            f"- 上次前缀指纹：{info.get('previous_fingerprint') or '无'}\n"
            f"- 本次前缀和上次是否一致：{self._format_prefix_same(info.get('prefix_same_as_previous'))}\n"
            f"- 真实请求前缀指纹：{info.get('actual_prefix_fingerprint') or '无'}\n"
            f"- 真实请求前缀是否一致：{self._format_prefix_same(info.get('actual_prefix_same_as_previous'))}\n"
            f"- 缓存键：{info.get('cache_key')}\n"
            f"- Session 缓存：{self._format_session_cache(info)}\n"
            f"- Session ID：{info.get('session_id_prefix') or '无'}\n"
            f"- Session 字段：{info.get('session_id_field') or '无'}\n"
            f"- 缓存命中依据：{info.get('session_cache_basis') or '服务端提示词缓存'}\n"
            f"- 稳定风格规则：{self._format_note(self._latest_style_rules)}\n"
            f"- 是否包装提供商方法：{self._format_bool(self._provider_wrapping_enabled())}\n"
            f"- 是否注入缓存字段：{self._format_bool(self._cache_injection_enabled())}\n"
            f"- 是否强制流式请求：{self._format_bool(self._openai_force_stream_enabled())}\n"
            f"- 前缀长度估算：{info.get('token_estimate')}\n"
            f"- 真实请求前缀估算：{info.get('actual_prefix_token_estimate') or 0}\n"
            f"- 首个动态内容位置估算：{self._format_dynamic_position(info)}\n"
            f"- 稳定前缀是否够长：{self._format_bool(self._prefix_meets_threshold(info))}\n"
            f"- OpenAI兼容门槛：{self._openai_threshold()}\n"
            f"- Claude门槛：{self._anthropic_threshold()}\n"
            f"- 接口是否在白名单：{self._format_bool(info.get('allowlisted'))}\n"
            f"- 白名单是否包含55系接口：{self._format_bool(self._allowlist_has_55ai())}\n"
            f"- 本次是否已注入：{self._format_bool(info.get('injected'))}\n"
            f"- 缓存键写入方式：{self._format_cache_key_mode()}\n"
            f"- 本轮已缓存 token：{self._format_cached_tokens(info)}\n"
            f"- Usage 状态：{self._format_usage_note(info)}\n"
            f"- 前缀风险：{self._format_risk_reasons(info)}\n"
            f"- 写入位置：{self._format_note(self._latest_write_target)}\n"
            f"- 缓存标记点数量：{getattr(self._latest_result, 'cache_breakpoints', 0) if self._latest_result else 0}\n"
            f"- 精确回复缓存：{self._format_note(self._latest_response_cache)}\n"
            f"- 是否发送保留时间字段：{self._format_bool(self._retention_enabled(info))}\n"
            f"- 备注：{self._format_note(info.get('note'))}"
        )

    def _format_bool(self, value: Any) -> str:
        return "是" if bool(value) else "否"

    def _format_session_cache(self, info: dict[str, Any]) -> str:
        if info.get("session_cache_enabled"):
            return "已启用"
        if str(info.get("base_url_host") or "").lower() == "aiwork.fans":
            return "未启用"
        return "不适用"

    def _format_note(self, value: Any) -> str:
        mapping = {
            "none": "无",
            "passthrough": "安全直通",
            "already_present_or_disabled": "已存在或未启用",
            "prepended": "已插入",
            "payload_prepended": "已插入到请求内容",
            "failed": "失败",
            "request observed": "已观察到请求",
            "cache injection disabled": "缓存注入未开启",
            "provider disabled": "该提供商未启用",
            "base_url not allowlisted": "接口地址不在白名单",
            "prefix below threshold": "稳定前缀长度低于门槛",
            "aiwork_session_id": "已发送 aiwork session_id",
            "prompt_cache_key+aiwork_session_id": "已发送 prompt_cache_key 和 aiwork session_id",
            "prompt_cache_key": "已发送缓存键 prompt_cache_key",
            "prompt_cache_key:near_threshold": "接近门槛，已发送缓存键 prompt_cache_key",
            "cache_control": "已写入 Claude 缓存标记",
            "no cacheable block": "没有可标记的缓存块",
            "implicit cache only": "仅使用隐式缓存",
            "cachedContent": "已引用 Gemini 显式缓存",
            "cache unavailable": "缓存不可用",
            "unsupported provider": "暂不支持该提供商",
            "disabled:prompt_cache_only": "已强制关闭，只使用服务端提示词缓存",
        }
        text = str(value or "")
        return mapping.get(text, text)

    def _format_prefix_same(self, value: Any) -> str:
        if value is None:
            return "首次记录，下一轮再判断"
        return "一致" if bool(value) else "不一致"

    def _format_cached_tokens(self, info: dict[str, Any]) -> str:
        value = info.get("observed_cached_tokens")
        if value is None:
            return "还没读到 usage"
        return str(value)

    def _format_usage_note(self, info: dict[str, Any]) -> str:
        note = str(info.get("usage_note") or "")
        if note == "observed":
            return "已读到 usage"
        if note == "usage not returned":
            return "上游本轮没有返回 usage，无法确认 cached_tokens"
        return "等待模型响应统计"

    def _format_cache_key_mode(self) -> str:
        if self.config.get("openai_cache_key_extra_body", False):
            return "顶层 + extra_body 实验写入"
        return "仅顶层"

    def _format_risk_reasons(self, info: dict[str, Any]) -> str:
        reasons = list(info.get("risk_reasons") or [])
        if info.get("front_not_static") and "front_not_static" not in reasons:
            reasons.append("front_not_static")
        if not reasons:
            return "未发现明显前缀风险"
        mapping = {
            "front_not_static": "请求开头不是固定 system/developer 内容",
            "dynamic_content_near_front": "时间/状态栏/检索摘要/动态记忆疑似太靠前",
            "media_near_front": "图片或 GIF 疑似太靠前",
        }
        text = "；".join(mapping.get(reason, str(reason)) for reason in reasons)
        if info.get("prefix_same_as_previous") is True:
            return f"风险提示，但本轮前缀指纹一致：{text}"
        return text

    def _format_cache_verdict(self, info: dict[str, Any]) -> str:
        if not info.get("allowlisted"):
            return "不会命中：接口地址不在白名单"
        if not self._provider_wrapping_enabled():
            return "不会命中：provider_wrapping_enabled 未开启"
        if not self._cache_injection_enabled():
            return "不会命中：cache_injection_enabled 未开启"
        if info.get("session_cache_enabled") and info.get("session_id_prefix"):
            if info.get("observed_cached_tokens") and int(info.get("observed_cached_tokens") or 0) > 0:
                return "已命中：后台返回了 cached_tokens"
            return "已发送 aiwork session_id：命中看站子 session cache"
        if not self._prefix_meets_threshold(info):
            return "难命中：稳定前缀还不够长"
        if info.get("injected") is not True:
            return f"未注入：{self._format_note(info.get('note'))}"
        if info.get("observed_cached_tokens") and int(info.get("observed_cached_tokens") or 0) > 0:
            return "已命中：后台返回了 cached_tokens"
        if info.get("actual_prefix_same_as_previous") is False:
            return "未稳定：真实请求前缀和上次不一致"
        if info.get("prefix_same_as_previous") is False:
            return "未稳定：本次前缀和上次不一致"
        if info.get("actual_prefix_same_as_previous") is True and info.get("usage_observed") and int(info.get("observed_cached_tokens") or 0) == 0:
            return "可能未透传：真实请求前缀一致但 cached_tokens 为 0，上游可能不支持或缓存偶发失效"
        if info.get("actual_prefix_same_as_previous") is True and info.get("usage_note") == "usage not returned":
            return "无法确认：真实请求前缀一致，但上游没有返回 usage"
        if info.get("actual_prefix_same_as_previous") is True:
            return "条件满足：真实请求前缀一致，等待后台返回 cached_tokens"
        if info.get("prefix_same_as_previous") is True:
            return "部分稳定：稳定前缀一致，下一轮继续看真实请求前缀"
        if info.get("front_not_static") or info.get("dynamic_prefix") or info.get("media_prefix"):
            return "有风险：动态内容或图片/GIF 靠前，下一轮如果变化会影响命中"
        return "首轮记录：下一轮相同前缀才好判断命中"

    def _format_dynamic_position(self, info: dict[str, Any]) -> str:
        value = info.get("first_dynamic_token_estimate")
        if value is None:
            return "未发现"
        try:
            token_pos = int(value)
        except (TypeError, ValueError):
            return str(value)
        if token_pos < 1024:
            return f"约 {token_pos} token，太靠前"
        return f"约 {token_pos} token，已在缓存门槛后"

    def _prefix_meets_threshold(self, info: dict[str, Any]) -> bool:
        provider = str(info.get("provider") or "")
        tokens = int(info.get("token_estimate") or 0)
        if provider == "anthropic":
            return tokens >= self._anthropic_threshold()
        if provider == "openai":
            return tokens >= max(0, self._openai_threshold() - self._threshold_slack("openai"))
        if provider == "gemini":
            return True
        return False

    def _threshold_slack(self, provider: str) -> int:
        try:
            return int(self.config.get("threshold_slack_tokens", {}).get(provider, 0) or 0)
        except Exception:
            return 0

    def _allowlist_has_55ai(self) -> bool:
        values = [*BUILTIN_ALLOWLIST_BASE_URLS, *list(self.config.get("allowlist_base_urls", []))]
        return any("api.55.ai" in str(value) or "api.55.al" in str(value) or str(value).strip() == "*" for value in values)

    def _anthropic_threshold(self) -> int:
        try:
            return int(self.config.get("min_prefix_tokens", {}).get("anthropic", 512))
        except Exception:
            return 512

    def _openai_threshold(self) -> int:
        try:
            return int(self.config.get("min_prefix_tokens", {}).get("openai", 1024))
        except Exception:
            return 1024

    def _retention_enabled(self, info: Optional[dict[str, Any]] = None) -> bool:
        retention = self.config.get("openai_prompt_cache_retention", {})
        if not isinstance(retention, dict) or not retention.get("enabled"):
            return False
        if info:
            class InspectInfo:
                base_url_host = str(info.get("base_url_host") or "")

            return openai_retention_allowed(InspectInfo(), self.config)
        return True

    def _provider_wrapping_enabled(self) -> bool:
        return bool(self.config.get("enabled", True) and self.config.get("provider_wrapping_enabled", False))

    def _cache_injection_enabled(self) -> bool:
        return bool(self.config.get("enabled", True) and self.config.get("cache_injection_enabled", False))

    def _openai_force_stream_enabled(self) -> bool:
        return bool(self.config.get("openai_force_stream", False))

    def _force_disable_exact_response_cache(self) -> None:
        exact = self.config.setdefault("exact_response_cache", {})
        if isinstance(exact, dict):
            exact["enabled"] = False
        self._latest_response_cache = "disabled:prompt_cache_only"

    def _apply_stable_style_rules_to_request(self, req: Any) -> None:
        current = getattr(req, "system_prompt", None)
        updated, inserted = with_stable_style_rules(current, self.config)
        if not inserted:
            self._latest_style_rules = "already_present_or_disabled"
            return
        try:
            setattr(req, "system_prompt", updated)
            self._latest_style_rules = "prepended"
        except Exception as exc:
            self._latest_style_rules = "failed"
            _log_warn(f"[PromptCacheMax] failed to prepend stable style rules: {exc}")

    def _infer_request_target(self, req: Any) -> tuple[str, str, str]:
        provider = self._first_attr(req, ("provider", "provider_type", "provider_id", "llm_provider")) or ""
        model = self._first_attr(req, ("model", "model_name", "model_id", "llm_model")) or ""
        base_url = self._first_attr(req, ("base_url", "api_base", "api_base_url", "openai_api_base", "api_url", "endpoint")) or ""
        if not provider and hasattr(self.context, "get_using_provider"):
            try:
                provider_obj = self.context.get_using_provider()
                provider = getattr(provider_obj, "provider_type", None) or provider_obj.__class__.__name__
                model = model or self._first_attr(provider_obj, ("model_name", "model", "model_id")) or ""
                base_url = base_url or self._provider_base_url(provider_obj)
            except Exception:
                pass
        return str(provider or ""), str(model or ""), str(base_url or "")

    def _first_attr(self, obj: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            value = getattr(obj, name, None)
            if value:
                return value
        if isinstance(obj, dict):
            for name in names:
                value = obj.get(name)
                if value:
                    return value
        return None

    def _provider_base_url(self, provider: Any) -> str:
        value = self._first_attr(
            provider,
            (
                "api_base",
                "base_url",
                "api_base_url",
                "openai_api_base",
                "api_url",
                "endpoint",
                "baseUrl",
            ),
        )
        if value:
            return str(value)
        config = self._first_attr(provider, ("config", "provider_config", "conf")) or {}
        return str(self._first_attr(config, ("api_base", "base_url", "api_base_url", "openai_api_base", "api_url", "endpoint")) or "")

    def _wrap_known_providers(self) -> None:
        providers = self._discover_provider_objects()
        for provider in providers:
            for method_name in (
                "_query",
                "_query_stream",
                "_prepare_query_config",
                "text_chat",
                "chat",
                "completion",
                "request",
                "_request",
                "_chat",
            ):
                method = getattr(provider, method_name, None)
                if method and inspect.iscoroutinefunction(method):
                    self._wrap_method(provider, method_name, method)
                elif method and method_name == "_prepare_query_config":
                    self._wrap_sync_method(provider, method_name, method)

    def _discover_provider_objects(self) -> list[Any]:
        objects: list[Any] = []
        for attr in ("get_using_provider", "provider", "provider_manager", "platform_manager"):
            obj = getattr(self.context, attr, None)
            if callable(obj):
                try:
                    result = obj()
                    if result:
                        objects.append(result)
                except Exception:
                    continue
            elif obj:
                objects.append(obj)
        expanded: list[Any] = []
        for obj in objects:
            expanded.append(obj)
            for attr in ("providers", "provider_insts", "provider_list"):
                items = getattr(obj, attr, None)
                if isinstance(items, dict):
                    expanded.extend(items.values())
                elif isinstance(items, (list, tuple)):
                    expanded.extend(items)
        seen = set()
        unique = []
        for obj in expanded:
            ident = id(obj)
            if ident not in seen:
                unique.append(obj)
                seen.add(ident)
        return unique

    def _wrap_method(self, provider: Any, method_name: str, original: Any) -> None:
        for obj, name, _original in self._wrapped:
            if obj is provider and name == method_name:
                return
        plugin = self

        async def wrapped(*args: Any, **kwargs: Any):
            provider_family = normalize_provider_with_config(
                getattr(provider, "provider_type", provider.__class__.__name__),
                getattr(provider, "model_name", "") or getattr(provider, "model", ""),
                plugin._provider_base_url(provider),
                plugin.config,
            )
            model = str(getattr(provider, "model_name", "") or getattr(provider, "model", "") or kwargs.get("model", ""))
            base_url = plugin._provider_base_url(provider)
            payload = plugin._extract_payload(args, kwargs)
            if apply_stable_style_rules_to_payload(payload, plugin.config):
                plugin._latest_style_rules = "payload_prepended"
            info = plugin._latest_info
            if info is None or info.provider != provider_family or (model and info.model and info.model != model):
                info = plugin._build_info_from_payload(provider_family, model, base_url, payload)
            risk_report = analyze_prefix_risks_from_payload(payload)
            plugin._latest_risk_report = risk_report
            if plugin._cache_injection_enabled():
                result = inject_payload(payload, info, plugin.config, plugin.state, plugin._create_gemini_cache)
            else:
                result = plugin._observe_only_result(payload, info)
            plugin._latest_result = result
            plugin.state.remember_inspect(
                info,
                result.injected,
                result.note,
                risk_report,
                getattr(result, "session_cache_enabled", False),
                getattr(result, "session_id_prefix", ""),
                getattr(result, "session_id_field", ""),
                getattr(result, "session_cache_basis", ""),
            )
            if result.injected:
                plugin._latest_write_target = plugin._write_payload(provider_family, args, kwargs, result.payload)
            else:
                plugin._latest_write_target = "none"
            plugin._latest_response_cache = "disabled:prompt_cache_only"
            _log_info(
                "[PromptCacheMax] "
                f"{provider_family}/{model} prefix={info.fingerprint[:12]} injected={result.injected} note={result.note}"
            )
            response = await original(*args, **kwargs)
            if response is not None:
                plugin._response_types[method_name] = response.__class__
            usage = plugin._extract_usage_from_response(response)
            if plugin.config.get("stats_enabled", True):
                plugin.state.record_usage(provider_family, model or info.model, usage)
            return response

        setattr(provider, method_name, wrapped)
        self._wrapped.append((provider, method_name, original))
        _log_info(f"[PromptCacheMax] wrapped provider {provider.__class__.__name__}.{method_name}")

    def _wrap_sync_method(self, provider: Any, method_name: str, original: Any) -> None:
        for obj, name, _original in self._wrapped:
            if obj is provider and name == method_name:
                return
        plugin = self

        def wrapped(*args: Any, **kwargs: Any):
            config = original(*args, **kwargs)
            result = plugin._latest_result
            if not result or not result.injected or result.provider != "gemini":
                return config
            cache_name = result.payload.get("cachedContent")
            if not cache_name:
                return config
            if isinstance(config, dict):
                config["cached_content"] = cache_name
                config["cachedContent"] = cache_name
                return config
            for attr in ("cached_content", "cachedContent"):
                try:
                    setattr(config, attr, cache_name)
                    return config
                except Exception:
                    continue
            return config

        setattr(provider, method_name, wrapped)
        self._wrapped.append((provider, method_name, original))
        _log_info(f"[PromptCacheMax] wrapped provider {provider.__class__.__name__}.{method_name}")

    def _extract_payload(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        for key in ("payload", "json", "body", "request_body"):
            value = kwargs.get(key)
            if isinstance(value, dict):
                return value
        for key in ("messages", "contents"):
            value = kwargs.get(key)
            if isinstance(value, list):
                return {key: value}
        for arg in args:
            if isinstance(arg, dict):
                return arg
            if isinstance(arg, list):
                return {"messages": arg}
        extra = kwargs.get("custom_extra_body")
        if isinstance(extra, dict):
            return extra
        return {}

    def _write_payload(self, provider: str, args: tuple[Any, ...], kwargs: dict[str, Any], payload: dict[str, Any]) -> str:
        for key in ("payload", "json", "body", "request_body"):
            if isinstance(kwargs.get(key), dict):
                kwargs[key].clear()
                kwargs[key].update(payload)
                return f"kwargs.{key}"
        if "messages" in payload and isinstance(kwargs.get("messages"), list):
            kwargs["messages"].clear()
            kwargs["messages"].extend(payload["messages"])
            return "kwargs.messages"
        if "contents" in payload and isinstance(kwargs.get("contents"), list):
            kwargs["contents"].clear()
            kwargs["contents"].extend(payload["contents"])
            return "kwargs.contents"
        for arg in args:
            if isinstance(arg, dict):
                arg.clear()
                arg.update(payload)
                return "args.dict"
            if isinstance(arg, list) and "messages" in payload:
                arg.clear()
                arg.extend(payload["messages"])
                return "args.messages"
            if isinstance(arg, list) and "contents" in payload:
                arg.clear()
                arg.extend(payload["contents"])
                return "args.contents"
        if provider == "openai":
            return "none:no_payload_target"
        extra_key = "custom_extra_body"
        extra = kwargs.setdefault(extra_key, {})
        if isinstance(extra, dict):
            for key, value in payload.items():
                if key not in ("messages", "contents"):
                    extra[key] = value
        if provider == "anthropic":
            custom_extra = kwargs.setdefault("custom_extra_body", {})
            if isinstance(custom_extra, dict):
                for key, value in payload.items():
                    if key not in ("messages", "contents"):
                        custom_extra[key] = value
        return extra_key

    def _observe_only_result(self, payload: dict[str, Any], info: Any):
        class Result:
            injected = False
            provider = info.provider
            fingerprint = info.fingerprint
            token_estimate = info.token_estimate
            note = "cache injection disabled"
            cache_breakpoints = 0

            def __init__(self, result_payload: dict[str, Any]):
                self.payload = result_payload

        return Result(payload)

    def _build_info_from_payload(self, provider: str, model: str, base_url: str, payload: dict[str, Any]):
        class PayloadReq:
            system_prompt = payload.get("system") or payload.get("system_instruction")
            tools = payload.get("tools")
            contexts = payload.get("messages") or payload.get("contents")

        return build_prefix_info(PayloadReq(), provider, model, base_url, self.config)

    def _create_gemini_cache(self, fingerprint: str, model: str, ttl_seconds: int) -> Optional[str]:
        provider = self._provider_for_family("gemini")
        if not provider:
            return None
        for method_name in ("create_cached_content", "create_cache", "_create_cached_content"):
            method = getattr(provider, method_name, None)
            if not method:
                continue
            try:
                result = method(model=model, ttl_seconds=ttl_seconds, display_name=f"pcache-{fingerprint[:12]}")
            except TypeError:
                try:
                    result = method(model, ttl_seconds)
                except Exception:
                    continue
            except Exception:
                continue
            if inspect.isawaitable(result):
                return None
            cache_name = self._extract_cache_name(result)
            if cache_name:
                return cache_name
        return None

    def _provider_for_family(self, family: str) -> Any:
        for provider in self._discover_provider_objects():
            provider_family = normalize_provider_with_config(
                getattr(provider, "provider_type", provider.__class__.__name__),
                getattr(provider, "model_name", "") or getattr(provider, "model", ""),
                self._provider_base_url(provider),
                self.config,
            )
            if provider_family == family:
                return provider
        return None

    def _extract_cache_name(self, result: Any) -> Optional[str]:
        if not result:
            return None
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("name") or result.get("cachedContent")
        return getattr(result, "name", None) or getattr(result, "cachedContent", None)

    def _extract_usage_from_response(self, response: Any) -> Any:
        if response is None:
            return {}
        for attr in ("usage", "raw_usage"):
            if hasattr(response, attr):
                return getattr(response, attr)
        for attr in ("raw_response", "response", "completion", "data"):
            if hasattr(response, attr):
                nested_usage = self._extract_usage_from_response(getattr(response, attr))
                if nested_usage:
                    return nested_usage
        if isinstance(response, dict):
            usage = response.get("usage") or response.get("usage_metadata") or response.get("usageMetadata") or {}
            if "input_cached" in response:
                usage = dict(usage) if isinstance(usage, dict) else {}
                usage["input_cached"] = response.get("input_cached")
            return usage
        return {}
