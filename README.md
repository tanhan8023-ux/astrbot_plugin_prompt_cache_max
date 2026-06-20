# AstrBot Prompt Cache Max

`astrbot_plugin_prompt_cache_max` 是一个偏保守的 AstrBot 提示词缓存辅助插件。
它的目标是帮助 OpenAI 兼容接口、Claude、Gemini 更容易吃到服务端 prompt cache。

插件不会复用模型回复，也不会主动改写用户语义。默认情况下，它处于安全模式：
不包装模型提供商、不注入缓存字段、不修改 system prompt。需要真正启用缓存注入时，
请手动打开对应开关。

## 命令

- `/pcache stats`：查看按提供商/模型聚合的缓存统计。
- `/pcache inspect`：查看最近一次缓存请求检查结果。
- `/pcache clear`：清除本地轻量状态和统计，不会删除聊天历史。

## 隐私

本地状态文件只保存指纹、提供商/模型名、缓存名、过期时间、估算 token 和计数器。
它不会保存原始 prompt、system prompt、工具内容或用户消息全文。

## 使用提示

未知 OpenAI 兼容接口默认不会收到专用缓存字段。确认上游支持后，再把接口地址加入
`allowlist_base_urls`，并手动开启缓存注入开关。

## aiwork.fans

`https://aiwork.fans/v1` 已作为 OpenAI 兼容接口加入默认白名单，但缓存注入仍然需要手动开启。
灰度测试时建议开启：

```json
{
  "observe_requests_enabled": true,
  "provider_wrapping_enabled": true,
  "cache_injection_enabled": true,
  "openai_prompt_cache_retention": {
    "enabled": false
  }
}
```

先用纯文字短消息确认 bot 正常回复，再观察 `/pcache inspect` 和 `/pcache stats`。
