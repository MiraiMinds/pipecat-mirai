# Pipecat docs submission

Files for the community-integration PR to [pipecat-ai/docs](https://github.com/pipecat-ai/docs):

1. `mirai.mdx` → `api-reference/server/services/tts/mirai.mdx`
2. A row in `api-reference/server/services/supported-services.mdx` (TTS table):
   `| [Mirai](/api-reference/server/services/tts/mirai) | `uv add pipecat-mirai` | Community |`
3. Register the page in `docs.json` navigation and add the redirect entry.
4. Attach a 30–60 s demo video showing a call and an interruption, then post in
   `#community-integrations` on the Pipecat Discord.

Submit after the repository is public and `pipecat-mirai` is on PyPI.
