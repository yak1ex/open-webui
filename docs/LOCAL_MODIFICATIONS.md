# Local features in `local/v0.12.5`

These customizations extend upstream Open WebUI 0.12.5.

## Image titles and prompt inspection

- Markdown image titles are shown as hover tooltips, including in the enlarged preview. Example: `![Fox](image-url "Watercolor fox")`.
- Generated chat images show their generation prompt on hover. When available, the tooltip also includes `Negative prompt: ...`.
- Prompt tooltips depend on the metadata returned with the image; older chat images may not have it.

## Built-in image generation

Use the built-in `generate_image` tool in a fresh chat with native tool calling and image generation enabled. The obsolete workspace tools **Image Gen Plus** and **Image Gen (0.5.3+)** are separate.

| Argument | Usage |
| --- | --- |
| `prompt` | Required description of the image. |
| `size` | Optional supported size, such as `1024x1024`. |
| `negative_prompt` | Optional elements to exclude; ignored by backends that do not support it. |
| `n` | Optional image count, at least 1. |
| `seed` | AUTOMATIC1111 only: `0`–`4294967295`, or `-1` for backend randomness. |

Omitted values preserve existing defaults. Explicit `n` and `seed` override the administrator's AUTOMATIC1111 batch-size and seed defaults. Supplying `seed` on another backend returns an error.

### Two ways to share a seed

**Separate calls:** obtain one random integer per image set, then reuse it as the actual `seed` argument in each `generate_image(n=1, ...)` call, varying the pose in each prompt. Obtain a fresh integer for the next set. No Dynamic Prompts configuration is required.

With code interpreter enabled, the LLM can call `execute_code` to run `import secrets; print(secrets.randbelow(2147483648))`, then use the returned integer in image calls within the same user turn. Browser-based Python execution requires the browser to remain connected.

**One batch:** enable Dynamic Prompts with **Fixed seed** using the administrator JSON below. Call `generate_image` with `n=2`, a seed, and a cyclical prompt such as:

```text
watercolor red fox, {@standing|sitting}, quiet green garden
```

The two images use the same seed and different poses. Without Fixed seed configuration, ordinary batches increment seeds. Seed reuse can help consistency but does not guarantee character identity.

### Optional administrator JSON

The current **AUTOMATIC1111 additional parameters** setting has been restored to `{}`. To enable same-seed Dynamic Prompts batches and suppress returned grid previews, merge:

```json
{
  "alwayson_scripts": {
    "Dynamic Prompts v2.17.1": {
      "args": [true, false, 1, false, false, false, 1.1, 1.5, 100, 0.7, true]
    }
  },
  "override_settings": {
    "return_grid": false
  }
}
```

This enables Dynamic Prompts and Fixed seed with combinatorial generation disabled. It does not set a global numeric seed: use a fresh seed for each new set. The script name must match the installed extension; the backend's `GET /sdapi/v1/scripts` lists it. Preserve other existing script and override entries when merging.

`return_grid: false` suppresses the extra returned grid preview; backend grid saving is separate. For separate-call workflows, keep only the `override_settings` entry if grid suppression is desired.

Depending on backend settings, `n=2` may return two generated images **plus a grid preview**. The preview does not count toward `n`; do not retry merely because it increases the returned file count.

## Legacy (non-native) tool calling

- Tool-call JSON accepts either `arguments` or the older `parameters` key; `parameters` takes precedence if both are present. The default legacy tool-selection prompt now requests `arguments`.
- The selected model's configured system prompt is included in the history supplied to legacy tool selection. Set that prompt on the model to carry its instructions into this step.
- These compatibility changes do not turn visible JSON into native tool calls. Use native mode for the built-in image workflows above.

