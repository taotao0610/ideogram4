# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Ideogram 4 is a 9.3B-parameter flow-matching Diffusion Transformer (DiT) for text-to-image generation, trained from scratch. The repo contains **inference code only** (no training code). Model weights are gated on Hugging Face under a non-commercial license.

## Commands

```bash
# Editable install (do this first)
pip install -e .

# Run inference
python run_inference.py --prompt "your prompt here"
python run_inference.py --prompt "..." --preset V4_TURBO_12           # fast preset (12 steps)
python run_inference.py --prompt "..." --preset V4_DEFAULT_20         # balanced (20 steps)
python run_inference.py --prompt "..." --no-magic-prompt              # skip LLM prompt expansion
python run_inference.py --prompt "..." --hive-api-key <key>           # enable safety screening

# Lint & type-check (after pip install pre-commit && pre-commit install)
pre-commit run --all-files
ruff check src/ run_inference.py
ruff format --check src/ run_inference.py
mypy src/ run_inference.py
```

No test suite is included in this open-source release.

## Architecture

**Inference pipeline** (`src/ideogram4/pipeline_ideogram4.py` → `Ideogram4Pipeline`):
1. **Text encoding** — Qwen3-VL-8B-Instruct extracts multi-layer hidden states (not just the final layer); this VLM replaces traditional CLIP/T5 encoders
2. **Magic prompt** — A plain-text `--prompt` is expanded to a structured JSON caption via an LLM (default: Ideogram's free hosted API; alternatives: Claude Opus/Sonnet via OpenRouter). Controlled by `src/ideogram4/magic_prompt.py`
3. **DiT backbone** — 34-layer `Ideogram4Transformer` (`src/ideogram4/modeling_ideogram4.py`) processes a joint text+image sequence with QK-RMSNorm, 3D MRoPE, SwiGLU MLP, and AdaLN
4. **Flow matching** — Euler integration over a logit-normal schedule (`src/ideogram4/scheduler.py`). Dual-branch classifier-free guidance (`guidance_scale` for prompt adherence, `mu` for image quality)
5. **VAE decoding** — KL autoencoder (`src/ideogram4/autoencoder.py`) decodes latents to pixel space

**Quantized weight loading** (`src/ideogram4/quantized_loading.py`):
- `nf4` (CUDA only, `ideogram-ai/ideogram-4-nf4`) — bitsandbytes 4-bit, default on CUDA
- `fp8` (cross-device, `ideogram-ai/ideogram-4-fp8`) — weight-only float8, default on MPS/CPU

**Sampler presets** (`src/ideogram4/sampler_configs.py`): `V4_QUALITY_48` (48 steps), `V4_DEFAULT_20` (20 steps), `V4_TURBO_12` (12 steps). Each preset switches `guidance_scale` from 7 to 3 for a final "polish" phase.

**Safety** (`src/ideogram4/safety.py`): Optional Hive text moderation (prompt screening) and visual moderation (output screening). Skipped silently when no API key is provided.

## Key constraints

- Model weights are **non-commercial** (`model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL`); the inference code is Apache 2.0
- Resolution must be a multiple of 16, in [256, 2048]; max aspect ratio 6:1
- The model expects **structured JSON captions** matching its training distribution; plain-text prompts without `--magic-prompt` may produce degraded results
