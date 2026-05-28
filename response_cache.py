from __future__ import annotations

import asyncio
import importlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from .cache_policy import stable_hash


@dataclass
class CacheLookup:
    hit: bool
    key: str
    value: Any = None
    backend: str = "memory"
    reason: str = ""


class ExactResponseCache:
    def __init__(self, config: dict[str, Any]):
        exact = config.get("exact_response_cache", {}) if isinstance(config, dict) else {}
        self.enabled = bool(exact.get("enabled", True))
        self.backend = str(exact.get("backend", "memory") or "memory").lower()
        self.redis_url = str(exact.get("redis_url", "redis://localhost:6379/0"))
        self.ttl_seconds = int(exact.get("ttl_seconds", 600) or 600)
        self.max_prompt_chars = int(exact.get("max_prompt_chars", 12000) or 12000)
        self._memory: dict[str, tuple[float, Any]] = {}
        self._redis = None

    def make_key(self, provider: str, model: str, base_url: str, payload: dict[str, Any]) -> Optional[str]:
        if not self.enabled:
            return None
        if _has_media_or_tool(payload):
            return None
        material = {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "payload": _stable_payload(payload),
        }
        raw = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(raw) > self.max_prompt_chars:
            return None
        return "pcache:exact:" + stable_hash(material)

    async def get(self, key: Optional[str]) -> CacheLookup:
        if not self.enabled or not key:
            return CacheLookup(False, key or "", backend=self.backend, reason="disabled_or_uncacheable")
        if self.backend == "redis":
            redis = await self._get_redis()
            if redis:
                raw = await redis.get(key)
                if raw:
                    return CacheLookup(True, key, json.loads(raw), "redis")
                return CacheLookup(False, key, backend="redis", reason="miss")
        item = self._memory.get(key)
        if not item:
            return CacheLookup(False, key, backend="memory", reason="miss")
        expires_at, value = item
        if expires_at < time.time():
            self._memory.pop(key, None)
            return CacheLookup(False, key, backend="memory", reason="expired")
        return CacheLookup(True, key, value, "memory")

    async def set(self, key: Optional[str], response: Any) -> bool:
        if not self.enabled or not key:
            return False
        value = _response_to_cache_value(response)
        if value is None:
            return False
        if self.backend == "redis":
            redis = await self._get_redis()
            if redis:
                await redis.setex(key, self.ttl_seconds, json.dumps(value, ensure_ascii=False))
                return True
        self._memory[key] = (time.time() + self.ttl_seconds, value)
        return True

    def restore(self, cached_value: Any, response_type: Any = None) -> Any:
        if isinstance(cached_value, dict) and response_type and cached_value.get("__class__"):
            kwargs = {
                key: value
                for key, value in cached_value.items()
                if not key.startswith("__")
            }
            return _restore_object(response_type, kwargs, cached_value)
        if isinstance(cached_value, dict) and cached_value.get("__class__") and cached_value.get("__module__"):
            try:
                module = importlib.import_module(str(cached_value["__module__"]))
                response_type = getattr(module, str(cached_value["__class__"]))
                kwargs = {
                    key: value
                    for key, value in cached_value.items()
                    if not key.startswith("__")
                }
                return _restore_object(response_type, kwargs, cached_value)
            except Exception:
                return cached_value
        return cached_value

    async def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            redis_mod = importlib.import_module("redis.asyncio")
            self._redis = redis_mod.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            return self._redis
        except Exception:
            self.backend = "memory"
            return None


def _stable_payload(payload: dict[str, Any]) -> Any:
    return _strip_cache_fields(payload)


def _strip_cache_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _strip_cache_fields(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if k not in ("cache_control", "prompt_cache_key", "cachedContent", "cached_content")
        }
    if isinstance(value, list):
        return [_strip_cache_fields(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _has_media_or_tool(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False, default=repr, separators=(",", ":")).lower()
    risky = ("image_url", "data:", "tool_calls", "function_call", "\"type\":\"image\"", "\"type\":\"tool\"")
    return any(marker in text for marker in risky)


def _response_to_cache_value(response: Any) -> Optional[Any]:
    if response is None:
        return None
    if isinstance(response, (str, dict, list, int, float, bool)):
        return response
    for attr in ("completion_text", "text", "content", "role", "usage"):
        if hasattr(response, attr):
            data = {
                "__class__": response.__class__.__name__,
                "__module__": response.__class__.__module__,
            }
            for name in ("completion_text", "text", "content", "role"):
                if hasattr(response, name):
                    data[name] = getattr(response, name)
            return data
    return None


def _restore_object(response_type: Any, kwargs: dict[str, Any], fallback: Any) -> Any:
    try:
        return response_type(**kwargs)
    except Exception:
        try:
            obj = response_type()
            for key, value in kwargs.items():
                setattr(obj, key, value)
            return obj
        except Exception:
            return fallback
