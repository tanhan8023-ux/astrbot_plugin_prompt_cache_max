# AstrBot Prompt Cache Max

`astrbot_plugin_prompt_cache_max` is a conservative AstrBot plugin that helps OpenAI,
Claude, and Gemini reuse provider-side prompt caches.

It does not reuse model responses and does not rewrite user meaning. It keeps stable
prefixes stable, injects provider cache hints where safe, and stores only lightweight
fingerprints and cache statistics.

By default it also prepends a stable warm, natural, lightly teasing style block to
the system prompt. This increases stable prefix length for prompt caching while
keeping exact response caching disabled, so replies can still vary naturally.

## Commands

- `/pcache stats` shows aggregate provider/model cache statistics.
- `/pcache inspect` shows the latest stable prefix fingerprint and provider capability.
- `/pcache clear` clears local lightweight cache state and statistics.

## Privacy

The state file stores fingerprints, provider/model names, cache names, expiry times,
token estimates, and counters. It intentionally does not store raw prompts, system
prompts, tools, or user messages.

## Notes

Unknown OpenAI-compatible endpoints are not sent provider-specific fields by default.
Add trusted endpoint prefixes to `allowlist_base_urls` if the upstream supports them.

## aiwork.fans

`https://aiwork.fans/v1` is included as an OpenAI-compatible endpoint, but cache
injection remains opt-in. To test it, enable `observe_requests_enabled`,
`provider_wrapping_enabled`, and `cache_injection_enabled`; keep
`openai_prompt_cache_retention.enabled` disabled unless the upstream documents
support for it.
