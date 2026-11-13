# Try 01 — Download TxGemma model weights

**Date:** 18-09-2026
**Script:** `download.sb`
**Job:** 22200876 (`hf_download`, COMPLETED)

## What it does
CPU job that runs `hf download google/txgemma-9b-predict` and
`hf download google/txgemma-27b-predict` into `$HF_HOME` (`/work/hdd/bisa/sshah21/hf_cache`).

## What happened
Completed successfully. Verified later in the investigation that the downloaded weights
are complete and correctly shaped (fp32 on disk, ~37GB for the 9B model, matching the
safetensors index; cast to bf16 at load time).
