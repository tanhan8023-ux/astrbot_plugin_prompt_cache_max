# AstrBot Prompt Cache Max

`astrbot_plugin_prompt_cache_max` 是一个偏保守的 AstrBot 提示词缓存辅助插件。
它的目标是帮助 OpenAI 兼容接口、Claude、Gemini 更容易吃到服务端 prompt cache。

插件不会复用模型回复，也不会主动改写用户语义。
只追求服务端 prompt cache / aiwork session cache 命中。默认情况下，它处于安全模式：
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
  ],
  "aiwork_session_cache_enabled": true
}
```

0.6.7 起，aiwork.fans 如果使用 Gemini 模型，会按 Gemini 的真实缓存门槛判断，而不是继续按 OpenAI 兼容接口的 1024 token 判断：`gemini-3` / `gemini-3.5` 按 4096 token，`gemini-2.5` 按 2048 token。开启缓存注入后，插件会自动把稳定缓存锚点补到门槛后面，`/pcache inspect` 会显示“实际缓存门槛”和“检测窗口”。

0.6.6 起，aiwork.fans 会在开启 `cache_injection_enabled` 后把稳定的 `session_id` 写入 HTTP Header。
不要把 `session_id` 写进请求体或 `extra_body`：Sub2API sticky session 读取的是 header，AstrBot 的 OpenAI 调用链里塞 `extra_body` 还会触发 SDK 参数重复。
这个 header 值由提供商、模型、接口域名和缓存键指纹生成，不包含用户原文、prompt 原文或聊天内容。
同一人设/同一模型/同一 aiwork 接口会保持稳定；换模型或换缓存键会变化。
插件不会为了 aiwork session cache 强制写入 `stream`、请求体 `session_id` 或 `extra_body`。

如果 `/pcache inspect` 里看到：

```text
Session 缓存：已启用
Session Header：session_id
Session 注入位置：HTTP header
Header 注入状态：已注入
缓存命中依据：Sub2API sticky session header
```

就说明插件已经把站子需要的 session cache 标识发出去了。
如果 header 已注入但仍不命中，请让站子确认前置 Nginx 已开启 `underscores_in_headers on;`，否则 `session_id` 这种带下划线的 header 会被代理层丢掉。

旧版只需要 prompt cache key 的最小配置仍可用：

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
如果 `/pcache inspect` 里的首个动态内容位置贴近“实际缓存门槛”，命中率会低。
当前版本在开启缓存注入时会自动追加固定“缓存稳定锚点”；aiwork + `gemini-3` 会垫到约 6144 token，aiwork + `gemini-2.5` 会垫到约 3072 token，
把动态时间、状态栏、最近聊天、图片/GIF 尽量挤到缓存窗口后面。
想提高命中率，重点看“真实请求前缀是否一致”和“首个动态内容位置估算”。
真实请求前缀要连续一致，首个动态内容必须出现在 inspect 显示的实际缓存门槛之后；`gemini-3` 建议放到 4096 token 之后，更稳妥是 6144 token 之后。
当前版本不会通过 payload 强制 `stream=true`，避免 AstrBot/OpenAI SDK 出现重复 stream 参数。
如果请求本身已经是流式，插件会补 `stream_options.include_usage=true`，用于让上游更容易返回 usage 和 `cached_tokens`。
