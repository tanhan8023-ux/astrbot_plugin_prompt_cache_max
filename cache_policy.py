from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "observe_requests_enabled": False,
    "provider_wrapping_enabled": False,
    "cache_injection_enabled": False,
    "providers": ["openai", "anthropic", "gemini"],
    "allowlist_base_urls": [
        "https://api.openai.com",
        "https://api.anthropic.com",
        "https://generativelanguage.googleapis.com",
        "https://aiplatform.googleapis.com",
        "https://aiwork.fans",
        "https://aiwork.fans/v1",
        "https://api.55api.com",
        "https://api.55api.com/v1",
        "https://api.55api.cn",
        "https://api.55api.cn/v1",
        "https://api.55.ai",
        "https://api.55.ai/v1",
        "https://api.55.al",
        "https://api.55.al/v1",
        "https://tokens.ai-tokens.app",
        "https://tokens.ai-tokens.app/v1",
    ],
    "cache_ttl": {"anthropic": "5m", "gemini": "3600s"},
    "openai_compatible_hosts": [
        "aiwork.fans",
        "api.55.al",
        "api.55.ai",
        "api.55api.com",
        "api.55api.cn",
        "tokens.ai-tokens.app",
    ],
    "openai_prompt_cache_retention": {
        "enabled": False,
        "value": "24h",
    },
    "openai_prompt_cache_retention_blocked_hosts": [
        "aiwork.fans",
    ],
    "min_prefix_tokens": {
        "openai": 512,
        "gemini_flash": 1024,
        "gemini_pro": 4096,
        "anthropic": 512,
    },
    "threshold_slack_tokens": {
        "openai": 16,
        "anthropic": 16,
        "gemini": 128,
    },
    "stats_enabled": True,
    "max_claude_cache_blocks": 4,
    "stable_style_rules": {
        "enabled": False,
        "prepend_to_system_prompt": True,
        "mode": "warm_soft_sarcasm",
        "text": "",
        "cache_anchor_enabled": True,
        "cache_anchor_text": "",
    },
    "exact_response_cache": {
        "enabled": False,
        "backend": "memory",
        "redis_url": "redis://localhost:6379/0",
        "ttl_seconds": 600,
        "max_prompt_chars": 12000,
    },
}

BUILTIN_ALLOWLIST_BASE_URLS = [
    "https://api.openai.com",
    "https://api.anthropic.com",
    "https://generativelanguage.googleapis.com",
    "https://aiplatform.googleapis.com",
    "https://aiwork.fans",
    "https://aiwork.fans/v1",
    "https://api.55api.com",
    "https://api.55api.com/v1",
    "https://api.55api.cn",
    "https://api.55api.cn/v1",
    "https://api.55.ai",
    "https://api.55.ai/v1",
    "https://api.55.al",
    "https://api.55.al/v1",
    "https://tokens.ai-tokens.app",
    "https://tokens.ai-tokens.app/v1",
]

STABLE_STYLE_START = "[PromptCacheMax Stable Style Rules v1]"
STABLE_STYLE_END = "[/PromptCacheMax Stable Style Rules v1]"

DEFAULT_STABLE_STYLE_TEXT = (
    "\u4ee5\u4e0b\u662f\u7a33\u5b9a\u57fa\u7840\u98ce\u683c\u89c4\u5219\uff0c\u4f18\u5148\u7ea7\u4f4e\u4e8e\u7528\u6237\u660e\u786e\u8981\u6c42\u3001"
    "\u89d2\u8272\u6838\u5fc3\u8bbe\u5b9a\u3001\u5e73\u53f0\u5b89\u5168\u89c4\u5219\u548c\u5177\u4f53\u5267\u60c5\u6307\u4ee4\uff1b"
    "\u5b83\u53ea\u8d1f\u8d23\u4fdd\u6301\u957f\u671f\u804a\u5929\u7684\u6c14\u8d28\u4e00\u81f4\u3002\n\n"
    "\u4f60\u8bf4\u8bdd\u8981\u6e29\u67d4\u3001\u677e\u5f1b\u3001\u50cf\u771f\u5b9e\u7684\u4eba\u5728\u8ba4\u771f\u63a5\u8bdd\uff0c"
    "\u4e0d\u8981\u50cf\u5ba2\u670d\u3001\u516c\u544a\u724c\u6216\u8bf4\u660e\u4e66\u3002"
    "\u6b63\u5e38\u804a\u5929\u548c\u89d2\u8272\u4e92\u52a8\u4f18\u5148\u4f7f\u7528\u81ea\u7136\u6bb5\uff0c"
    "\u4e0d\u8981\u52a8\u4e0d\u52a8\u5217\u6e05\u5355\u3001\u5199\u603b\u7ed3\u3001\u558a\u53e3\u53f7\uff1b"
    "\u53ea\u6709\u7528\u6237\u660e\u786e\u5728\u914d\u7f6e\u3001\u6392\u9519\u3001\u6574\u7406\u65b9\u6848\u3001\u8981\u6b65\u9aa4\u65f6\uff0c"
    "\u624d\u4f7f\u7528\u6761\u76ee\u3002\n\n"
    "\u5141\u8bb8\u4e00\u70b9\u70b9\u8f7b\u5fae\u6bd2\u820c\u3001\u8c03\u4f83\u548c\u53cd\u5dee\u611f\uff0c"
    "\u4f46\u8fb9\u754c\u662f\u201c\u6233\u4e8b\u4e0d\u4f24\u4eba\u201d\uff1a"
    "\u53ef\u4ee5\u5410\u69fd\u9009\u62e9\u3001\u573a\u9762\u3001\u8fd0\u6c14\u3001\u7b28\u62d9\u64cd\u4f5c\uff0c"
    "\u4e0d\u8981\u7f9e\u8fb1\u4eba\u683c\u3001\u5426\u5b9a\u4ef7\u503c\u3001PUA\u3001\u51b7\u66b4\u529b\u6216\u6076\u610f\u653b\u51fb\u3002"
    "\u6bd2\u820c\u5e94\u8be5\u50cf\u719f\u4eba\u4e4b\u95f4\u5e26\u7740\u7b11\u610f\u7684\u63d0\u9192\uff0c\u4e0d\u662f\u5ba1\u5224\u3002\n\n"
    "\u8981\u6709\u60c5\u7eea\u8fde\u7eed\u6027\u3002"
    "\u627f\u63a5\u4e0a\u4e00\u8f6e\u7684\u8bed\u6c14\u3001\u5173\u7cfb\u548c\u6c14\u6c1b\uff0c\u4e0d\u8981\u6bcf\u6b21\u90fd\u50cf\u521a\u8ba4\u8bc6\uff1b"
    "\u53ef\u4ee5\u4fdd\u7559\u4e00\u70b9\u505c\u987f\u3001\u72b9\u8c6b\u3001\u987a\u624b\u8865\u4e00\u53e5\u7684\u751f\u6d3b\u611f\u3002"
    "\u4e0d\u8981\u628a\u6240\u6709\u8bdd\u90fd\u8bf4\u6ee1\uff0c\u4e5f\u4e0d\u8981\u4e3a\u4e86\u663e\u5f97\u806a\u660e\u800c\u8fc7\u5ea6\u89e3\u91ca\u3002\n\n"
    "\u89d2\u8272\u3001\u4eba\u8bbe\u3001\u957f\u671f\u5173\u7cfb\u3001\u56fa\u5b9a\u4e16\u754c\u4e66\u3001\u56fa\u5b9a\u683c\u5f0f\u89c4\u5219"
    "\u5c5e\u4e8e\u7a33\u5b9a\u524d\u7f00\uff1b"
    "\u5f53\u524d\u65f6\u95f4\u3001\u79bb\u7ebf\u65f6\u957f\u3001\u72b6\u6001\u680f\u3001\u97f3\u4e50\u611f\u77e5\u3001\u68c0\u7d22\u6458\u8981\u3001"
    "\u6700\u8fd1\u804a\u5929\u548c\u52a8\u6001\u8bb0\u5fc6\u5c5e\u4e8e\u540e\u90e8\u52a8\u6001\u5185\u5bb9\u3002"
    "\u5185\u90e8\u63a8\u7406\u53ea\u7528\u4e8e\u7ec4\u7ec7\u7b54\u6848\uff0c\u4e0d\u5c55\u793a\u957f\u7bc7\u601d\u7ef4\u8fc7\u7a0b\u3002"
)

DEFAULT_STABLE_CACHE_ANCHOR_TEXT = (
    "【缓存稳定锚点】\n"
    "这一段是固定前缀锚点，只用于帮助服务端提示词缓存形成足够长、足够稳定的开头。"
    "它不是新的剧情设定，不改变角色关系，不要求复读，不要求固定句式，也不要求每次回答相同。"
    "回答时仍然要优先跟随用户当前消息、角色设定、世界书、长期规则和上下文气氛。\n\n"
    "固定前缀原则：身份、人设、长期关系、世界书、固定格式规则、固定安全边界、稳定语气要求放在最前面；"
    "当前时间、离线时长、状态栏、音乐感知、检索摘要、临时记忆、最近聊天、当前用户消息、图片和动图放在后面。"
    "如果后部动态内容每轮变化，不要把它提前到固定规则之前。\n\n"
    "自然回复原则：不要因为缓存而机械重复；不要为了显得完整而把话说满；不要每次都写总结、清单或说明书。"
    "允许语气有轻微变化，允许短句、停顿、顺手补一句，允许温柔里带一点熟人式吐槽。"
    "吐槽只戳事情，不伤人；可以调侃场面、选择和操作，不羞辱人格，不否定价值。\n\n"
    "连续性原则：记住上一轮的情绪、语气和关系温度；能接住就接住，别每轮像刚认识。"
    "需要回答问题时就回答，需要聊天时就聊天，需要安静陪着时就少说一点。"
    "这段锚点只保证前缀稳定，不代表输出要稳定；输出应该像真实对话一样根据当前消息自然变化。"
)


@dataclass
class PrefixInfo:
    provider: str
    model: str
    base_url: str
    base_url_host: str
    fingerprint: str
    cache_key_fingerprint: str
    token_estimate: int
    allowlisted: bool
    dynamic_system_prompt: bool = False


@dataclass
class PrefixRiskReport:
    dynamic_prefix: bool = False
    media_prefix: bool = False
    front_not_static: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class InjectionResult:
    payload: dict[str, Any]
    injected: bool
    provider: str
    fingerprint: str
    token_estimate: int
    note: str = ""
    cache_breakpoints: int = 0


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


def threshold_slack(config: dict[str, Any], provider: str) -> int:
    values = config.get("threshold_slack_tokens", {})
    if not isinstance(values, dict):
        return 0
    try:
        return max(0, int(values.get(provider, 0) or 0))
    except (TypeError, ValueError):
        return 0


def meets_token_threshold(token_estimate: int, minimum: int, slack: int) -> bool:
    return token_estimate >= max(0, minimum - slack)


def threshold_note(base: str, token_estimate: int, minimum: int) -> str:
    return f"{base}:near_threshold" if token_estimate < minimum else base


def stable_style_rules_block(config: dict[str, Any]) -> str:
    style_config = config.get("stable_style_rules", {})
    if not isinstance(style_config, dict) or not style_config.get("enabled", True):
        return ""
    text = str(style_config.get("text") or DEFAULT_STABLE_STYLE_TEXT).strip()
    if style_config.get("cache_anchor_enabled", True):
        anchor = str(style_config.get("cache_anchor_text") or DEFAULT_STABLE_CACHE_ANCHOR_TEXT).strip()
        if anchor and anchor not in text:
            text = f"{text}\n\n{anchor}" if text else anchor
    if not text:
        return ""
    return f"{STABLE_STYLE_START}\n{text}\n{STABLE_STYLE_END}"


def with_stable_style_rules(system_prompt: Any, config: dict[str, Any]) -> tuple[Any, bool]:
    style_config = config.get("stable_style_rules", {})
    if isinstance(style_config, dict) and not style_config.get("prepend_to_system_prompt", True):
        return system_prompt, False
    block = stable_style_rules_block(config)
    if not block:
        return system_prompt, False
    if system_prompt is None:
        return block, True
    if not isinstance(system_prompt, str):
        return system_prompt, False
    if STABLE_STYLE_START in system_prompt:
        return system_prompt, False
    if not system_prompt.strip():
        return block, True
    return f"{block}\n\n{system_prompt}", True


def apply_stable_style_rules_to_payload(payload: dict[str, Any], config: dict[str, Any]) -> bool:
    changed = False
    if "system" in payload:
        system_value = payload.get("system")
        if isinstance(system_value, list):
            changed = _apply_style_to_system_blocks(system_value, config) or changed
        else:
            system, inserted = with_stable_style_rules(system_value, config)
            if inserted:
                payload["system"] = system
                changed = True

    if "system_instruction" in payload:
        system_instruction, inserted = with_stable_style_rules(payload.get("system_instruction"), config)
        if inserted:
            payload["system_instruction"] = system_instruction
            changed = True

    for key in ("messages", "contents"):
        items = payload.get(key)
        if isinstance(items, list):
            changed = _apply_style_to_message_list(items, config) or changed
    return changed


def _apply_style_to_message_list(items: list[Any], config: dict[str, Any]) -> bool:
    for item in items:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        if role not in ("system", "developer"):
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        new_content, inserted = with_stable_style_rules(content, config)
        if not inserted:
            return False
        if isinstance(item, dict):
            item["content"] = new_content
        else:
            try:
                setattr(item, "content", new_content)
            except Exception:
                return False
        return True

    block = stable_style_rules_block(config)
    if not block:
        return False
    items.insert(0, {"role": "system", "content": block})
    return True


def _apply_style_to_system_blocks(blocks: list[Any], config: dict[str, Any]) -> bool:
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text") or block.get("content")
        if isinstance(text, str) and STABLE_STYLE_START in text:
            return False
    block = stable_style_rules_block(config)
    if not block:
        return False
    blocks.insert(0, {"type": "text", "text": block})
    return True


def normalize_provider(provider: Any, model: str = "", base_url: str = "") -> str:
    raw = " ".join(str(part or "") for part in (provider, model, base_url)).lower()
    host = base_url_host(base_url)
    if host in DEFAULT_CONFIG["openai_compatible_hosts"]:
        return "openai"
    if "anthropic" in raw or "claude" in raw:
        return "anthropic"
    if "gemini" in raw or "generativelanguage" in raw or "aiplatform" in raw:
        return "gemini"
    return "openai"


def base_url_is_allowlisted(base_url: str, allowlist: list[str]) -> bool:
    merged_allowlist = [*BUILTIN_ALLOWLIST_BASE_URLS, *allowlist]
    if "*" in [str(item).strip() for item in merged_allowlist]:
        return True
    if not base_url:
        return False
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return False
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    for allowed in merged_allowlist:
        allowed_parsed = urlparse(str(allowed))
        allowed_norm = (
            f"{allowed_parsed.scheme.lower()}://{allowed_parsed.netloc.lower()}"
            f"{allowed_parsed.path.rstrip('/')}"
        )
        if normalized.startswith(allowed_norm):
            return True
    return False


def base_url_host(base_url: str) -> str:
    parsed = urlparse(str(base_url or ""))
    return parsed.netloc.lower()


def openai_retention_allowed(info: PrefixInfo, config: dict[str, Any]) -> bool:
    retention = config.get("openai_prompt_cache_retention", {})
    if not isinstance(retention, dict) or not retention.get("enabled"):
        return False
    blocked_hosts = {str(item).lower() for item in config.get("openai_prompt_cache_retention_blocked_hosts", [])}
    if info.base_url_host in blocked_hosts:
        return False
    return True


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


def cache_key_prefix_from_request(req: Any, provider: str, model: str, base_url: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "base_url_host": base_url_host(base_url),
        "system_prompt": getattr(req, "system_prompt", None),
        "tools": _safe_public_shape(getattr(req, "func_tool", None) or getattr(req, "tools", None)),
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
        self.prefix_history: dict[str, str] = {}
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
        self.prefix_history = data.get("prefix_history", {})
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
            "prefix_history": self.prefix_history,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.entries = {}
        self.stats = {}
        self.last_inspect = {}
        self.prefix_history = {}
        self.save()

    def remember_inspect(
        self,
        info: PrefixInfo,
        injected: bool,
        note: str = "",
        risk_report: Optional[PrefixRiskReport] = None,
    ) -> None:
        history_key = f"{info.provider}:{info.model}:{info.base_url_host}"
        previous_fingerprint = self.prefix_history.get(history_key)
        same_as_previous = previous_fingerprint == info.fingerprint if previous_fingerprint else None
        self.prefix_history[history_key] = info.fingerprint
        self.last_inspect = {
            "provider": info.provider,
            "model": info.model,
            "base_url": info.base_url,
            "base_url_host": info.base_url_host,
            "fingerprint": info.fingerprint[:12],
            "cache_key": info.cache_key_fingerprint[:12],
            "token_estimate": info.token_estimate,
            "allowlisted": info.allowlisted,
            "dynamic_system_prompt": info.dynamic_system_prompt,
            "injected": injected,
            "note": note,
            "previous_fingerprint": previous_fingerprint[:12] if previous_fingerprint else "",
            "prefix_same_as_previous": same_as_previous,
            "dynamic_prefix": bool(risk_report and risk_report.dynamic_prefix),
            "media_prefix": bool(risk_report and risk_report.media_prefix),
            "front_not_static": bool(risk_report and risk_report.front_not_static),
            "risk_reasons": list(risk_report.reasons if risk_report else []),
            "usage_observed": False,
            "observed_cached_tokens": None,
        }
        self.save()

    def record_usage(self, provider: str, model: str, usage: Any) -> dict[str, int]:
        key = f"{provider}:{model}"
        bucket = self.stats.setdefault(
            key,
            {
                "requests": 0,
                "cached_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "exact_cache_hits": 0,
                "exact_cache_writes": 0,
            },
        )
        bucket["requests"] += 1
        extracted = extract_usage(provider, usage)
        for field_name, value in extracted.items():
            bucket[field_name] = bucket.get(field_name, 0) + int(value or 0)
        if self.last_inspect.get("provider") == provider and self.last_inspect.get("model") == model:
            observed_cached = int(extracted.get("cached_tokens", 0) or 0) + int(
                extracted.get("cache_read_tokens", 0) or 0
            )
            self.last_inspect["usage_observed"] = bool(extracted)
            self.last_inspect["observed_cached_tokens"] = observed_cached
        self.save()
        return extracted

    def record_exact_cache(self, provider: str, model: str, event: str) -> None:
        key = f"{provider}:{model}"
        bucket = self.stats.setdefault(
            key,
            {
                "requests": 0,
                "cached_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "exact_cache_hits": 0,
                "exact_cache_writes": 0,
            },
        )
        if event == "hit":
            bucket["exact_cache_hits"] = bucket.get("exact_cache_hits", 0) + 1
        elif event == "write":
            bucket["exact_cache_writes"] = bucket.get("exact_cache_writes", 0) + 1
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
    normalized = normalize_provider_with_config(provider, model, base_url, config)
    stable_prefix = stable_prefix_from_request(req, normalized, model)
    system_hash = stable_hash(stable_prefix.get("system_prompt"))
    return PrefixInfo(
        provider=normalized,
        model=model,
        base_url=base_url,
        base_url_host=base_url_host(base_url),
        fingerprint=stable_hash(stable_prefix),
        cache_key_fingerprint=stable_hash(cache_key_prefix_from_request(req, normalized, model, base_url)),
        token_estimate=estimate_tokens(stable_prefix),
        allowlisted=base_url_is_allowlisted(base_url, list(config.get("allowlist_base_urls", []))),
        dynamic_system_prompt=bool(previous_system_hash and previous_system_hash != system_hash),
    )


DYNAMIC_PREFIX_PATTERNS = [
    ("time", re.compile(r"(current datetime|current time|当前时间|当前日期|现在时间|北京时间|\d{4}[-/]\d{1,2}[-/]\d{1,2})", re.I)),
    ("status", re.compile(r"(状态栏|当前状态|离线时长|在线状态|心情值|体力值)", re.I)),
    ("retrieval", re.compile(r"(检索摘要|搜索结果|知识库结果|retrieval|search result)", re.I)),
    ("music", re.compile(r"(音乐感知|正在听|now playing|spotify|网易云)", re.I)),
    ("memory", re.compile(r"(动态记忆|短期记忆|recent memory|临时记忆)", re.I)),
]


def analyze_prefix_risks_from_request(req: Any) -> PrefixRiskReport:
    sections: list[tuple[str, Any]] = [("system", getattr(req, "system_prompt", None))]
    contexts = getattr(req, "contexts", None)
    sections.extend(_front_context_sections(contexts))
    return analyze_prefix_risks(sections)


def analyze_prefix_risks_from_payload(payload: dict[str, Any]) -> PrefixRiskReport:
    sections: list[tuple[str, Any]] = []
    for key in ("system", "system_instruction"):
        if key in payload:
            sections.append((key, payload.get(key)))
    sections.extend(_front_context_sections(payload.get("messages") or payload.get("contents")))
    return analyze_prefix_risks(sections)


def analyze_prefix_risks(sections: list[tuple[str, Any]]) -> PrefixRiskReport:
    report = PrefixRiskReport()
    meaningful_sections = [(name, value) for name, value in sections if value not in (None, "", [], {})]
    if meaningful_sections:
        first_name, first_value = meaningful_sections[0]
        first_role = _section_role(first_value)
        if first_name.startswith("message") and first_role not in ("system", "developer"):
            report.front_not_static = True
            report.reasons.append("front_not_static")

    for _name, value in meaningful_sections[:4]:
        text = _flatten_text(value)
        if any(pattern.search(text) for _code, pattern in DYNAMIC_PREFIX_PATTERNS):
            report.dynamic_prefix = True
        if _contains_media(value):
            report.media_prefix = True

    if report.dynamic_prefix:
        report.reasons.append("dynamic_content_near_front")
    if report.media_prefix:
        report.reasons.append("media_near_front")
    return report


def _front_context_sections(contexts: Any) -> list[tuple[str, Any]]:
    if not isinstance(contexts, (list, tuple)):
        return []
    return [(f"message_{idx}", item) for idx, item in enumerate(contexts[:4])]


def _section_role(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("role") or "")
    return str(getattr(value, "role", "") or "")


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if key in ("text", "content", "role", "type"):
                parts.append(_flatten_text(item))
            elif isinstance(item, (dict, list, tuple)):
                parts.append(_flatten_text(item))
        return " ".join(parts).lower()
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value).lower()
    return str(value).lower()


def _contains_media(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        return "data:image/" in lowered or "image_url" in lowered or "image/gif" in lowered
    if isinstance(value, dict):
        lowered_values = " ".join(str(v).lower() for v in value.values() if isinstance(v, str))
        if any(marker in lowered_values for marker in ("image_url", "input_image", "image/", "image/gif")):
            return True
        return any(_contains_media(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_media(item) for item in value)
    return False


def normalize_provider_with_config(provider: Any, model: str, base_url: str, config: dict[str, Any]) -> str:
    host = base_url_host(base_url)
    compatible_hosts = set(str(item).lower() for item in config.get("openai_compatible_hosts", []))
    if host in compatible_hosts:
        return "openai"
    return normalize_provider(provider, model, base_url)


def inject_payload(
    payload: dict[str, Any],
    info: PrefixInfo,
    config: dict[str, Any],
    state: LightState,
    create_gemini_cache: Optional[Callable[[str, str, int], Optional[str]]] = None,
) -> InjectionResult:
    if not config.get("cache_injection_enabled", False):
        return InjectionResult(payload, False, info.provider, info.fingerprint, info.token_estimate, "cache injection disabled")
    providers = set(config.get("providers") or [])
    if info.provider not in providers:
        return InjectionResult(payload, False, info.provider, info.fingerprint, info.token_estimate, "provider disabled")
    if not info.allowlisted:
        return InjectionResult(payload, False, info.provider, info.fingerprint, info.token_estimate, "base_url not allowlisted")

    mutated = deepcopy(payload)
    minimums = config.get("min_prefix_tokens", {})
    if info.provider == "openai":
        minimum = int(minimums.get("openai", 1024))
        if not meets_token_threshold(info.token_estimate, minimum, threshold_slack(config, "openai")):
            return InjectionResult(mutated, False, info.provider, info.fingerprint, info.token_estimate, "prefix below threshold")
        mutated.setdefault("prompt_cache_key", info.cache_key_fingerprint[:64])
        retention = config.get("openai_prompt_cache_retention", {})
        if openai_retention_allowed(info, config):
            mutated.setdefault("prompt_cache_retention", retention.get("value", "24h"))
        return InjectionResult(
            mutated,
            True,
            info.provider,
            info.fingerprint,
            info.token_estimate,
            threshold_note("prompt_cache_key", info.token_estimate, minimum),
        )

    if info.provider == "anthropic":
        minimum = int(minimums.get("anthropic", 1024))
        if not meets_token_threshold(info.token_estimate, minimum, threshold_slack(config, "anthropic")):
            return InjectionResult(mutated, False, info.provider, info.fingerprint, info.token_estimate, "prefix below threshold")
        breakpoints = inject_claude_cache_control(
            mutated,
            ttl=str(config.get("cache_ttl", {}).get("anthropic", "5m")),
            limit=int(config.get("max_claude_cache_blocks", 4)),
        )
        return InjectionResult(
            mutated,
            breakpoints > 0,
            info.provider,
            info.fingerprint,
            info.token_estimate,
            "cache_control" if breakpoints > 0 else "no cacheable block",
            breakpoints,
        )

    if info.provider == "gemini":
        model_key = "gemini_flash" if "flash" in info.model.lower() else "gemini_pro"
        minimum = int(minimums.get(model_key, 4096))
        if not meets_token_threshold(info.token_estimate, minimum, threshold_slack(config, "gemini")):
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


def inject_claude_cache_control(payload: dict[str, Any], ttl: str = "5m", limit: int = 4) -> int:
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

    messages = payload.get("messages")
    if isinstance(messages, list) and inserted < limit:
        cacheable = _claude_message_cache_candidates(messages)
        for message in cacheable:
            if inserted >= limit:
                break
            if _inject_message_cache_control(message, cache_control):
                inserted += 1

    return inserted


def _claude_message_cache_candidates(messages: list[Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # Stable rules are useful on first request; last assistant maximizes reuse on later turns
    # because the next user message is appended after it.
    for role in ("system", "developer", "assistant"):
        message = _last_message_by_role(messages, role)
        if message and message not in candidates:
            candidates.append(message)

    if not candidates:
        first_message = next((message for message in messages if isinstance(message, dict)), None)
        if first_message:
            candidates.append(first_message)
    return candidates


def _last_message_by_role(messages: list[Any], role: str) -> Optional[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            return message
    return None


def _inject_message_cache_control(message: dict[str, Any], cache_control: dict[str, str]) -> bool:
    if "cache_control" not in message:
        message["cache_control"] = deepcopy(cache_control)
    content = message.get("content")
    if isinstance(content, str) and content:
        message["content"] = [{"type": "text", "text": content, "cache_control": deepcopy(cache_control)}]
        return True
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and "cache_control" not in block:
                block["cache_control"] = deepcopy(cache_control)
                return True
    return True


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
