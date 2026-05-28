from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from .cache_policy import (
    BUILTIN_ALLOWLIST_BASE_URLS,
    LightState,
    build_prefix_info,
    inject_payload,
    merge_config,
    normalize_provider,
    stable_hash,
)

PLUGIN_VERSION = "0.1.8"

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
    "Maximize provider-side prompt cache reuse for OpenAI, Claude, and Gemini.",
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
        if self.config.get("enabled", True):
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
        self._wrap_known_providers()
        provider, model, base_url = self._infer_request_target(req)
        provider_family = normalize_provider(provider, model, base_url)
        key = f"{provider_family}:{model}"
        previous_system_hash = self._last_system_hash_by_key.get(key)
        system_hash = stable_hash(getattr(req, "system_prompt", None))
        info = build_prefix_info(req, provider_family, model, base_url, self.config, previous_system_hash)
        self._last_system_hash_by_key[key] = system_hash
        self._latest_info = info
        self.state.remember_inspect(info, False, "request observed")
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
            yield event.plain_result("Prompt cache lightweight state cleared.")
            return
        yield event.plain_result("Usage: /pcache stats | inspect | clear")

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
            return "No prompt cache stats recorded yet."
        lines = ["Prompt cache stats:"]
        for key, stat in sorted(self.state.stats.items()):
            requests = stat.get("requests", 0)
            cached = stat.get("cached_tokens", 0)
            read = stat.get("cache_read_tokens", 0)
            created = stat.get("cache_creation_tokens", 0)
            lines.append(
                f"- {key}: requests={requests}, cached={cached}, "
                f"read={read}, created={created}"
            )
        return "\n".join(lines)

    def _format_inspect(self) -> str:
        info = self.state.last_inspect
        if not info:
            return "No prompt cache request has been observed yet."
        return (
            "Last prompt cache request:\n"
            f"- plugin_version: {PLUGIN_VERSION}\n"
            f"- provider/model: {info.get('provider')}/{info.get('model')}\n"
            f"- base_url: {info.get('base_url')}\n"
            f"- base_url_host: {info.get('base_url_host')}\n"
            f"- fingerprint: {info.get('fingerprint')}\n"
            f"- token_estimate: {info.get('token_estimate')}\n"
            f"- anthropic_threshold: {self._anthropic_threshold()}\n"
            f"- allowlisted: {info.get('allowlisted')}\n"
            f"- allowlist_has_55ai: {self._allowlist_has_55ai()}\n"
            f"- injected: {info.get('injected')}\n"
            f"- note: {info.get('note')}"
        )

    def _allowlist_has_55ai(self) -> bool:
        values = [*BUILTIN_ALLOWLIST_BASE_URLS, *list(self.config.get("allowlist_base_urls", []))]
        return any("api.55.ai" in str(value) or "api.55.al" in str(value) or str(value).strip() == "*" for value in values)

    def _anthropic_threshold(self) -> int:
        try:
            return int(self.config.get("min_prefix_tokens", {}).get("anthropic", 512))
        except Exception:
            return 512

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
            provider_family = normalize_provider(
                getattr(provider, "provider_type", provider.__class__.__name__),
                getattr(provider, "model_name", "") or getattr(provider, "model", ""),
                plugin._provider_base_url(provider),
            )
            model = str(getattr(provider, "model_name", "") or getattr(provider, "model", "") or kwargs.get("model", ""))
            base_url = plugin._provider_base_url(provider)
            payload = plugin._extract_payload(args, kwargs)
            info = plugin._latest_info
            if info is None or info.provider != provider_family or (model and info.model and info.model != model):
                info = plugin._build_info_from_payload(provider_family, model, base_url, payload)
            result = inject_payload(payload, info, plugin.config, plugin.state, plugin._create_gemini_cache)
            plugin._latest_result = result
            plugin.state.remember_inspect(info, result.injected, result.note)
            if result.injected:
                plugin._write_payload(provider_family, args, kwargs, result.payload)
            _log_info(
                "[PromptCacheMax] "
                f"{provider_family}/{model} prefix={info.fingerprint[:12]} injected={result.injected} note={result.note}"
            )
            response = await original(*args, **kwargs)
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

    def _write_payload(self, provider: str, args: tuple[Any, ...], kwargs: dict[str, Any], payload: dict[str, Any]) -> None:
        for key in ("payload", "json", "body", "request_body"):
            if isinstance(kwargs.get(key), dict):
                kwargs[key].clear()
                kwargs[key].update(payload)
                return
        if "messages" in payload and isinstance(kwargs.get("messages"), list):
            kwargs["messages"].clear()
            kwargs["messages"].extend(payload["messages"])
        if "contents" in payload and isinstance(kwargs.get("contents"), list):
            kwargs["contents"].clear()
            kwargs["contents"].extend(payload["contents"])
        for arg in args:
            if isinstance(arg, dict):
                arg.clear()
                arg.update(payload)
                return
        extra_key = "extra_body" if provider == "openai" else "custom_extra_body"
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
            provider_family = normalize_provider(
                getattr(provider, "provider_type", provider.__class__.__name__),
                getattr(provider, "model_name", "") or getattr(provider, "model", ""),
                self._provider_base_url(provider),
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
        if isinstance(response, dict):
            usage = response.get("usage") or response.get("usage_metadata") or response.get("usageMetadata") or {}
            if "input_cached" in response:
                usage = dict(usage) if isinstance(usage, dict) else {}
                usage["input_cached"] = response.get("input_cached")
            return usage
        return {}
