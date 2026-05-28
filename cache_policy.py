from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "providers": ["openai", "anthropic", "gemini"],
    "allowlist_base_urls": [
        "https://api.openai.com",
        "https://api.anthropic.com",
        "https://generativelanguage.googleapis.com",
        "https://aiplatform.googleapis.com",
        "https://api.55api.com",
        "https://api.55api.com/v1",
        "https://api.55api.cn",
        "https://api.55api.cn/v1",
    ],
    "cache_ttl": {"anthropic": "5m", "gemini": "3600s"},
    "min_prefix_tokens": {
        "openai": 1024,
        "gemini_flash": 1024,
        "gemini_pro": 4096,
        "anthropic": 1024,
    },
    "stats_enabled": True,
    "max_claude_cache_blocks": 4,
}


@dataclass
class PrefixInfo:
    provider: str
    model: str
    base_url: str
    fingerprint: str
    token_estimate: int
    allowlisted: bool
    dynamic_system_prompt: bool = False


@dataclass
class InjectionResult:
    payload: dict[str, Any]
    injected: bool
    provider: str
    fingerprint: str
    token_estimate: int
    note: str = ""


@dataclass
class CacheEntry:
    provider: str
    model: str
    cache_name: Optional[str] = None
    expires_at: float = 0
    token_estimate: int = 0
    stats: dict[str, int] = field(default_factory=dict)


def merge_config(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_CONFIG)
    if not config:
        return merged
    if not isinstance(config, dict):
        config = {
            key: config.get(key)  # type: ignore[attr-defined]
            for key in DEFAULT_CONFIG
            if hasattr(config, "get") and config.get(key) is not None  # type: ignore[attr-defined]
        }
    for key, value in config.items():
        if key == "allowlist_base_urls" and isinstance(value, list):
            seen = set()
            merged[key] = [
                url
                for url in [*DEFAULT_CONFIG["allowlist_base_urls"], *value]
                if not (url in seen or seen.add(url))
            ]
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def estimate_tokens(value: Any) -> int:
    text = canonical_json(value) if not isinstance(value, str) else value
    if not text:
        return 0
    return max(1, len(text) // 4)


def normalize_provider(provider: Any, model: str = "", base_url: str = "") -> str:
    raw = " ".join(str(part or "") for part in (provider, model, base_url)).lower()
    if "anthropic" in raw or "claude" in raw:
        return "anthropic"
    if "gemini" in raw or "generativelanguage" in raw or "aiplatform" in raw:
        return "gemini"
    return "openai"


def base_url_is_allowlisted(base_url: str, allowlist: list[str]) -> bool:
    if "*" in [str(item).strip() for item in allowlist]:
        return True
    if not base_url:
        return False
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return False
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    for allowed in allowlist:
        allowed_parsed = urlparse(str(allowed))
        allowed_norm = (
            f"{allowed_parsed.scheme.lower()}://{allowed_parsed.netloc.lower()}"
            f"{allowed_parsed.path.rstrip('/')}"
        )
        if normalized.startswith(allowed_norm):
            return True
    return False


def parse_ttl_seconds(value: Any, default: int) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    try:
        if raw.endswith("ms"):
            return max(1, int(float(raw[:-2]) / 1000))
        if raw.endswith("s"):
            return int(float(raw[:-1]))
        if raw.endswith("m"):
            return int(float(raw[:-1]) * 60)
        if raw.endswith("h"):
            return int(float(raw[:-1]) * 3600)
        return int(float(raw))
    except ValueError:
        return default


def stable_prefix_from_request(req: Any, provider: str, model: str) -> dict[str, Any]:
    system_prompt = getattr(req, "system_prompt", None)
    tools = getattr(req, "func_tool", None) or getattr(req, "tools", None)
    conversation = getattr(req, "conversation", None)
    contexts = getattr(req, "contexts", None)
    return {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "tools": _safe_public_shape(tools),
        "conversation": _safe_public_shape(conversation),
        "contexts": _stable_context_shape(contexts),
    }


def _safe_public_shape(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_public_shape(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_safe_public_shape(v) for v in value]
    public = {}
    for name in ("name", "description", "parameters", "schema", "type"):
        if hasattr(value, name):
            public[name] = _safe_public_shape(getattr(value, name))
    return public or repr(type(value).__name__)


def _stable_context_shape(contexts: Any) -> Any:
    if not contexts:
        return None
    stable_items = []
    for item in contexts if isinstance(contexts, (list, tuple)) else [contexts]:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if role in ("system", "developer", "tool"):
            stable_items.append({"role": role, "content": content})
    return stable_items or None


class LightState:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, CacheEntry] = {}
        self.stats: dict[str, dict[str, int]] = {}
        self.last_inspect: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.stats = data.get("stats", {})
        self.last_inspect = data.get("last_inspect", {})
        for fingerprint, raw in data.get("entries", {}).items():
            self.entries[fingerprint] = CacheEntry(
                provider=str(raw.get("provider", "")),
                model=str(raw.get("model", "")),
                cache_name=raw.get("cache_name"),
                expires_at=float(raw.get("expires_at", 0) or 0),
                token_estimate=int(raw.get("token_estimate", 0) or 0),
                stats={str(k): int(v) for k, v in raw.get("stats", {}).items()},
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": {
                key: {
                    "provider": entry.provider,
                    "model": entry.model,
                    "cache_name": entry.cache_name,
                    "expires_at": entry.expires_at,
                    "token_estimate": entry.token_estimate,
                    "stats": entry.stats,
                }
                for key, entry in self.entries.items()
            },
            "stats": self.stats,
            "last_inspect": self.last_inspect,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.entries = {}
        self.stats = {}
        self.last_inspect = {}
        self.save()

    def remember_inspect(self, info: PrefixInfo, injected: bool, note: str = "") -> None:
        self.last_inspect = {
            "provider": info.provider,
            "model": info.model,
            "base_url": info.base_url,
            "fingerprint": info.fingerprint[:12],
            "token_estimate": info.token_estimate,
            "allowlisted": info.allowlisted,
            "dynamic_system_prompt": info.dynamic_system_prompt,
            "injected": injected,
            "note": note,
        }
        self.save()

    def record_usage(self, provider: str, model: str, usage: Any) -> None:
        key = f"{provider}:{model}"
        bucket = self.stats.setdefault(
            key,
            {
                "requests": 0,
                "cached_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
            },
        )
        bucket["requests"] += 1
        extracted = extract_usage(provider, usage)
        for field_name, value in extracted.items():
            bucket[field_name] = bucket.get(field_name, 0) + int(value or 0)
        self.save()

    def get_or_create_gemini_cache(
        self,
        fingerprint: str,
        provider: str,
        model: str,
        token_estimate: int,
        ttl_seconds: int,
        create_cache: Optional[Callable[[str, str, int], Optional[str]]] = None,
    ) -> Optional[str]:
        now = time.time()
        existing = self.entries.get(fingerprint)
        if existing and existing.cache_name and existing.expires_at > now:
            return existing.cache_name
        cache_name = create_cache(fingerprint, model, ttl_seconds) if create_cache else None
        if not cache_name:
            return None
        self.entries[fingerprint] = CacheEntry(
            provider=provider,
            model=model,
            cache_name=cache_name,
            expires_at=now + ttl_seconds,
            token_estimate=token_estimate,
        )
        self.save()
        return cache_name


def build_prefix_info(
    req: Any,
    provider: str,
    model: str,
    base_url: str,
    config: dict[str, Any],
    previous_system_hash: Optional[str] = None,
) -> PrefixInfo:
    normalized = normalize_provider(provider, model, base_url)
    stable_prefix = stable_prefix_from_request(req, normalized, model)
    system_hash = stable_hash(stable_prefix.get("system_prompt"))
    return PrefixInfo(
        provider=normalized,
        model=model,
        base_url=base_url,
        fingerprint=stable_hash(stable_prefix),
        token_estimate=estimate_tokens(stable_prefix),
        allowlisted=base_url_is_allowlisted(base_url, list(config.get("allowlist_base_urls", []))),
        dynamic_system_prompt=bool(previous_system_hash and previous_system_hash != system_hash),
    )


def inject_payload(
    payload: dict[str, Any],
    info: PrefixInfo,
    config: dict[str, Any],
    state: LightState,
    create_gemini_cache: Optional[Callable[[str, str, int], Optional[str]]] = None,
) -> InjectionResult:
    providers = set(config.get("providers") or [])
    if info.provider not in providers:
        return InjectionResult(payload, False, info.provider, info.fingerprint, info.token_estimate, "provider disabled")
    if not info.allowlisted:
        return InjectionResult(payload, False, info.provider, info.fingerprint, info.token_estimate, "base_url not allowlisted")

    mutated = deepcopy(payload)
    minimums = config.get("min_prefix_tokens", {})
    if info.provider == "openai":
        minimum = int(minimums.get("openai", 1024))
        if info.token_estimate < minimum:
            return InjectionResult(mutated, False, info.provider, info.fingerprint, info.token_estimate, "prefix below threshold")
        mutated.setdefault("prompt_cache_key", info.fingerprint[:64])
        return InjectionResult(mutated, True, info.provider, info.fingerprint, info.token_estimate, "prompt_cache_key")

    if info.provider == "anthropic":
        minimum = int(minimums.get("anthropic", 1024))
        if info.token_estimate < minimum:
            return InjectionResult(mutated, False, info.provider, info.fingerprint, info.token_estimate, "prefix below threshold")
        injected = inject_claude_cache_control(
            mutated,
            ttl=str(config.get("cache_ttl", {}).get("anthropic", "5m")),
            limit=int(config.get("max_claude_cache_blocks", 4)),
        )
        return InjectionResult(mutated, injected, info.provider, info.fingerprint, info.token_estimate, "cache_control" if injected else "no cacheable block")

    if info.provider == "gemini":
        model_key = "gemini_flash" if "flash" in info.model.lower() else "gemini_pro"
        minimum = int(minimums.get(model_key, 4096))
        if info.token_estimate < minimum:
            return InjectionResult(mutated, False, info.provider, info.fingerprint, info.token_estimate, "implicit cache only")
        ttl = parse_ttl_seconds(config.get("cache_ttl", {}).get("gemini"), 3600)
        cache_name = state.get_or_create_gemini_cache(
            info.fingerprint,
            info.provider,
            info.model,
            info.token_estimate,
            ttl,
            create_gemini_cache,
        )
        if cache_name:
            mutated["cachedContent"] = cache_name
            return InjectionResult(mutated, True, info.provider, info.fingerprint, info.token_estimate, "cachedContent")
        return InjectionResult(mutated, False, info.provider, info.fingerprint, info.token_estimate, "cache unavailable")

    return InjectionResult(mutated, False, info.provider, info.fingerprint, info.token_estimate, "unsupported provider")


def inject_claude_cache_control(payload: dict[str, Any], ttl: str = "5m", limit: int = 4) -> bool:
    inserted = 0
    cache_control = {"type": "ephemeral", "ttl": ttl}

    system = payload.get("system")
    if isinstance(system, str) and system:
        payload["system"] = [{"type": "text", "text": system, "cache_control": deepcopy(cache_control)}]
        inserted += 1
    elif isinstance(system, list):
        for block in reversed(system):
            if inserted >= limit:
                break
            if isinstance(block, dict) and "cache_control" not in block:
                block["cache_control"] = deepcopy(cache_control)
                inserted += 1
                break

    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if inserted >= limit:
                break
            if isinstance(tool, dict) and "cache_control" not in tool:
                tool["cache_control"] = deepcopy(cache_control)
                inserted += 1

    return inserted > 0


def extract_usage(provider: str, usage: Any) -> dict[str, int]:
    data = _to_mapping(usage)
    if not data:
        return {}
    if "input_cached" in data:
        return {"cached_tokens": int(data.get("input_cached", 0) or 0)}
    if provider == "openai":
        prompt_details = _to_mapping(data.get("prompt_tokens_details"))
        return {"cached_tokens": int(prompt_details.get("cached_tokens", 0) or 0)}
    if provider == "anthropic":
        return {
            "cache_read_tokens": int(data.get("cache_read_input_tokens", 0) or 0),
            "cache_creation_tokens": int(data.get("cache_creation_input_tokens", 0) or 0),
        }
    if provider == "gemini":
        return {"cached_tokens": int(data.get("cached_content_token_count", 0) or data.get("cachedContentTokenCount", 0) or 0)}
    return {}


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    result = {}
    for name in (
        "prompt_tokens_details",
        "cached_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cached_content_token_count",
        "cachedContentTokenCount",
        "input_cached",
    ):
        if hasattr(value, name):
            result[name] = getattr(value, name)
    return result
