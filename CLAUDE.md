# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Inference code for **Ideogram 4**, a 9.3B-parameter flow-matching DiT text-to-image model. This repo contains **inference only** — no training code, no test suite, no benchmark runner.

- **Two-tier license**: code is Apache 2.0 ([LICENSE.md](LICENSE.md)); model weights are non-commercial ([model_licenses/](model_licenses/)). Redistribution/deployment obligations differ.
- **Gated weights**: first download requires `HF_TOKEN` and accepting the license on Hugging Face.
- **Two weight repos**: `ideogram-ai/ideogram-4-nf4` (CUDA only, Diffusers-compatible) and `ideogram-ai/ideogram-4-fp8` (any device). The CLI picks one via `--quantization`.

## Commands

```bash
# Install (editable, for development)
pip install -e .

# Lint + format + typecheck in one shot (canonical entry point)
pre-commit run --all-files

# Manual equivalents
ruff check --fix src run_inference.py
ruff format src run_inference.py
mypy src run_inference.py

# Run inference
python run_inference.py \
  --prompt "a ginger cat in a wizard hat" \
  --output output.png \
  --sampler-preset V4_QUALITY_48 \
  --quantization nf4 \
  --magic-prompt-key "$IDEOGRAM_API_KEY"
```

**No test suite exists** — do not assume `pytest` or similar commands are available.

### Local launchers (gitignored except `run_gemma.zsh`)

[start.zsh](start.zsh) (`V4_QUALITY_48`) and [start2.zsh](start2.zsh) (`V4_TURBO_12`) assume the conda env `ideogram4` is activated and use `deepseek-v1` as the magic-prompt model. [run_gemma.zsh](run_gemma.zsh) runs the experimental VLM script under `src/diffusiongemma/`.

## Code Style

- **2-space Python indentation** (`pyproject.toml` sets `indent-width = 2`, `indent-style = "space"`). This is unusual for Python — preserve it when editing existing files.
- Line length 120; formatter is **ruff, not Black**.
- mypy is scoped to `src/` and `run_inference.py`; missing imports are ignored for `transformers.*`, `huggingface_hub.*`, `bitsandbytes.*`, `requests.*`.

## Environment Variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Required to download the gated model weights. |
| `IDEOGRAM_API_KEY` | Default key for the `ideogram-4-v1` magic-prompt backend. |
| `MAGIC_PROMPT_API_KEY` | Override read first by `--magic-prompt-key`; used for OpenRouter (Claude) and DeepSeek backends. |
| `HIVE_TEXT_MODERATION_KEY` | Prompt NSFW/hate/violence screening. **Missing → silently skipped with a warning.** |
| `HIVE_VISUAL_MODERATION_KEY` | Output image screening. **Missing → silently skipped with a warning.** |

## Architecture

### Entry point and flow ([run_inference.py](run_inference.py))

`main()` runs in this order: Hive text moderation → magic prompt expansion (optional) → `Ideogram4Pipeline.from_pretrained(...)` → `pipe(...)` → Hive visual moderation → save.

Defaults are device-aware: `_default_device()` returns `cuda → mps → cpu`; `_default_quantization()` returns `nf4` on CUDA, else `fp8`. CLI flag authority is `run_inference.py`'s argparse — note the current names are `--sampler-preset`, `--hive-text-key`, `--hive-visual-key`, `--magic-prompt-model`, `--magic-prompt-key` (older docs may use stale names).

### Package layout ([src/ideogram4/](src/ideogram4/))

- [pipeline_ideogram4.py](src/ideogram4/pipeline_ideogram4.py) — `Ideogram4Pipeline.from_pretrained` and `__call__`: shard-aware weight download, Qwen3-VL text encoding, Euler flow-matching denoising, VAE decode.
- [modeling_ideogram4.py](src/ideogram4/modeling_ideogram4.py) — DiT backbone (`emb_dim=4608`, 34 transformer blocks).
- [quantized_loading.py](src/ideogram4/quantized_loading.py) — bnb NF4 and weight-only FP8 loaders (see *Quantization* below).
- [magic_prompt.py](src/ideogram4/magic_prompt.py) — LLM-backed prompt expansion (see *Magic Prompt* below).
- [caption_verifier.py](src/ideogram4/caption_verifier.py) — enforces the canonical JSON caption schema; `--warn-on-caption-issues` downgrades errors to warnings.
- [sampler_configs.py](src/ideogram4/sampler_configs.py) — `PRESETS` registry (see *Sampler presets* below).
- [scheduler.py](src/ideogram4/scheduler.py) — `LogitNormalSchedule` and Euler step helpers.
- [autoencoder.py](src/ideogram4/autoencoder.py), [latent_norm.py](src/ideogram4/latent_norm.py), [constants.py](src/ideogram4/constants.py) — VAE, per-channel latent norm constants, and key constants (`QWEN3_VL_ACTIVATION_LAYERS`, `IMAGE_POSITION_OFFSET=65536`, `LLM_TOKEN_INDICATOR=3`, `OUTPUT_IMAGE_INDICATOR=2`).

### Pipeline `__call__` data flow

1. `_verify_prompts` runs `CaptionVerifier` (raise or warn per `--warn-on-caption-issues`).
2. Build `LogitNormalSchedule` from resolution; compute step intervals via `make_step_intervals(num_steps)`.
3. `_build_inputs` packs `[pad | text tokens | image tokens]`; builds 3D position ids `(t, h, w)` (text mirrored across all three axes; image offset by `IMAGE_POSITION_OFFSET=65536`), `segment_ids`, and `indicator`.
4. `_encode_text` walks Qwen3-VL `language_model.layers`, taps hidden states at the **13 layers in `QWEN3_VL_ACTIVATION_LAYERS`**, and stacks+concatenates them as conditioning. Non-LLM positions are zeroed.
5. **Asymmetric CFG**: the negative branch is **image-only** (sliced past `max_text_tokens`) with zeroed `neg_llm_features`. This is not standard CFG — do not "fix" it.
6. Euler flow-matching loop iterates **from `num_steps-1` down to 0**: `v = gw_i * pos_v + (1 - gw_i) * neg_v`; `z = z + v * (s_val - t_val)`.
7. `_decode`: per-channel latent norm → unpatch → `autoencoder.decoder` → clamp `[-1,1]` → uint8 PIL.

### Quantization ([src/ideogram4/quantized_loading.py](src/ideogram4/quantized_loading.py))

- **NF4 (CUDA only)**: loaded via bitsandbytes `Params4bit.from_prequantized`. `_build_transformer` raises `ValueError` on non-CUDA devices — this is why the CLI defaults to `fp8` on MPS/CPU.
- **FP8 e4m3 weight-only (any device)**: `Fp8Linear` keeps fp8 weight + fp32 `weight_scale` **on CPU**; `forward` dequantizes via `.to(x.dtype)` and runs a bf16 matmul. No FP8 tensor cores required.
- **MPS compatibility (commit `e2abdb9`)**: MPS cannot store `float8_e4m3fn`, so `load_fp8_state_dict` pins fp8 weights + scales to CPU; `Fp8Linear.forward` explicitly moves only the dequantized bf16 weight onto the device per call (`if w.device != x.device: w = w.to(x.device)`). When editing the FP8 path, preserve this device separation.

### Magic Prompt ([src/ideogram4/magic_prompt.py](src/ideogram4/magic_prompt.py))

`MagicPrompt.expand(prompt, aspect_ratio) -> str` returns a minified JSON caption. Three underlying helpers:

- `openrouter_chat` — OpenAI-compatible POST to `https://openrouter.ai/api/v1/chat/completions` (used by the Claude backends).
- `anthropic_chat` — converts OpenAI-style messages to Anthropic protocol (`system` hoisted to top level, `thinking: {type: "disabled"}`). **Reads `content[].text`, falling back to `content[].thinking`** when only thinking blocks are returned — this is the compatibility path for reasoning models like DeepSeek (commit `e2abdb9`).
- `ideogram_magic_prompt` — Ideogram's hosted magic-prompt API.

The `MAGIC_PROMPTS` registry at [magic_prompt.py:488](src/ideogram4/magic_prompt.py#L488) maps: `ideogram-4-v1` (default, `DEFAULT_MAGIC_PROMPT`), `claude-sonnet-v1`, `claude-opus-v1`, `deepseek-v1` (uses `anthropic_chat` against `https://api.deepseek.com/anthropic`, model `deepseek-v4-pro`).

**JSON parsing recovery (commit `8a5a58a`)**: `strip_aspect_ratio_and_bboxes` first tries `json.loads`; on `JSONDecodeError` it dumps the bad caption to `faile_log/bad_caption_<timestamp>.txt` (the typo `faile_log` matches the hardcoded string and the tracked directory), then falls back to `_extract_first_json` (uses `json.JSONDecoder().raw_decode` from the first `{`, handling LLM prose prefixes and trailing footnotes). Do not remove this fallback when touching the parser.

System prompt templates live in [magic_prompt_system_prompts/](src/ideogram4/magic_prompt_system_prompts/) as `[NAME]`-sectioned `.txt` files with `{{aspect_ratio}}` / `{{original_prompt}}` placeholders.

### Sampler presets ([src/ideogram4/sampler_configs.py](src/ideogram4/sampler_configs.py))

Three presets in `PRESETS` at [sampler_configs.py:10](src/ideogram4/sampler_configs.py#L10): `V4_QUALITY_48` (default), `V4_DEFAULT_20`, `V4_TURBO_12`. Each bundles `num_steps`, per-step `guidance_schedule`, `mu`, `std`.

**Critical trap**: `guidance_schedule` is stored in **loop-index order, where index 0 is the final (polish) step** — the reverse of intuition. The typical pattern is N cleanup steps at `gw=3` followed by N main steps at `gw=7`. When editing or adding presets, write the schedule in loop-index order.

## Constraints

- Resolution must be a **multiple of 16**, range **256–2048**, max aspect ratio **6:1**.
- Captions should be structured JSON (fields like `colour_palette`, `compositional_decomposition`); `CaptionVerifier` enforces the schema.
- Magic prompt is **on by default** — pass `--no-magic-prompt` only when feeding an already-structured caption.

## Local Additions (not in upstream)

- [src/diffusiongemma/](src/diffusiongemma/) — experimental, **not installed by `pyproject.toml`**. Depends on `mlx_vlm` (Apple Silicon only). Entry [diffusiongemma-26B-A4B-it-4bit.py](src/diffusiongemma/diffusiongemma-26B-A4B-it-4bit.py) is a standalone script that loads `mlx-community/diffusiongemma-26B-A4B-it-4bit` and describes a hardcoded `input/girl.jpg` in Chinese.
- [faile_log/](faile_log/) — where malformed magic-prompt captions are dumped (the typo is intentional — it matches the hardcoded path in `magic_prompt.py`).
- [output/](output/) and [input/](input/) — gitignored working directories for generated images and sample inputs.

## Authoritative Sources (when in doubt)

- CLI args: `python run_inference.py --help`
- Quantization strategy: [quantized_loading.py](src/ideogram4/quantized_loading.py) + `_build_transformer` in [pipeline_ideogram4.py](src/ideogram4/pipeline_ideogram4.py)
- Magic prompt providers: the `MAGIC_PROMPTS` dict in [magic_prompt.py](src/ideogram4/magic_prompt.py)
- Sampler presets: the `PRESETS` dict in [sampler_configs.py](src/ideogram4/sampler_configs.py)
- Dev setup and CLI reference: [docs/development.md](docs/development.md) and [docs/inference.md](docs/inference.md)
