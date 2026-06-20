# AstrBot Prompt Cache Max

`astrbot_plugin_prompt_cache_max` 是一个偏保守的 AstrBot 提示词缓存辅助插件。
它的目标是帮助 OpenAI 兼容接口、Claude、Gemini 更容易吃到服务端 prompt cache。

插件不会复用模型回复，也不会主动改写用户语义。当前版本会强制关闭精确回复缓存，
只追求服务端 prompt cache 命中，也就是后台 usage 里的 `cached_tokens` 增长。默认情况下，它处于安全模式：
不包装模型提供商、不注入缓存字段、不修改 system prompt。需要真正启用缓存注入时，
请手动打开对应开关。

## 命令

- `/pcache stats`：查看按提供商/模型聚合的缓存统计。
- `/pcache inspect`：查看最近一次缓存请求检查结果。
- `/pcache clear`：清除本地轻量状态和统计，不会删除聊天历史。

`/pcache inspect` 会直接用中文提示：接口是否白名单、是否注入 `prompt_cache_key`、
稳定前缀是否够长、本次前缀和上次是否一致，以及动态时间、状态栏、图片/GIF 是否太靠前。

## 隐私

本地状态文件只保存指纹、提供商/模型名、缓存名、过期时间、估算 token 和计数器。
它不会保存原始 prompt、system prompt、工具内容或用户消息全文。

## 使用提示

未知 OpenAI 兼容接口默认不会收到专用缓存字段。确认上游支持后，再把接口地址加入
`allowlist_base_urls`，并手动开启缓存注入开关。

## aiwork.fans

`https://aiwork.fans/v1` 已作为 OpenAI 兼容接口加入默认白名单，但缓存注入仍然需要手动开启。
灰度测试时建议开启。aiwork 默认只发送 `prompt_cache_key`，不发送 `prompt_cache_retention`：

```json
{
  "observe_requests_enabled": true,
  "provider_wrapping_enabled": true,
  "cache_injection_enabled": true,
  "openai_prompt_cache_retention": {
    "enabled": false
  },
  "openai_prompt_cache_retention_blocked_hosts": [
    "aiwork.fans"
  ]
}
```

先用纯文字短消息确认 bot 正常回复，再观察 `/pcache inspect` 和 `/pcache stats`。

真正命中缓存要满足“前缀完全一致”：固定人设、世界书、长期规则放最前面；
当前时间、离线时长、状态栏、音乐感知、当前消息、最近聊天、图片/GIF 放后面。
如果 `/pcache inspect` 里的前缀长度贴近 1024，建议开启 `stable_style_rules.enabled`，
当前版本会追加固定“缓存稳定锚点”，把固定前缀垫得更长一点，避免动态时间或状态栏挤进前 1024 tokens。
想提高命中率，重点看“真实请求前缀是否一致”和“首个动态内容位置估算”。
真实请求前缀要连续一致，首个动态内容最好出现在 1024 token 之后；更稳妥是 1536 token 之后。
