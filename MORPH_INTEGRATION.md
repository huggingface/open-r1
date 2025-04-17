# Morph Integration for Open-R1

This document describes the integration of [Morph](https://cloud.morph.so/web/) as an alternative code execution provider for Open-R1.

## Code Changes

The implementation is organized into two main areas of functionality:

### 1. General Code Execution (Codeforces/GRPO Training)

**New Files:**
- `src/open_r1/utils/routed_morph.py` - Implementation of Morph execution for general code snippets
- `scripts/morph_router.py` - Router service for Morph execution
- `slurm/morph_router.slurm` - Slurm script for launching the Morph router

**Modified Files:**
- `src/open_r1/configs.py` - Added configuration options:
  - `morph_router_url` - URL for the Morph router service
  - `provider_type` option that accepts "morph" value
- `src/open_r1/rewards.py` - Added Morph support to code execution reward functions
- `src/open_r1/utils/code_providers.py` - Added Morph as a code execution provider option
- `README.md` - Added documentation for using Morph for general code execution

**Configuration Example:**
```yaml
# For regular code execution in GRPO training
provider_type: morph
morph_router_url: "your-server-ip:8001"
```

### 2. IOI Problem Execution

**New Files:**
- `src/open_r1/utils/ioi/morph_client.py` - Client for executing IOI problems with Morph

**Modified Files:**
- `src/open_r1/configs.py` - Added configuration options:
  - `ioi_provider` option that accepts "morph" value
- `src/open_r1/rewards.py` - Added Morph support to IOI reward functions
- `README.md` - Added documentation for using Morph with IOI problems

**Configuration Example:**
```yaml
# For IOI problems
ioi_provider: morph
```

## Dependencies

To use the Morph provider, users need to:
1. Install the `morphcloud` package: `pip install morphcloud`
2. Obtain a Morph API key and set it in `.env`: `MORPH_API_KEY="your_key_here"`

## Language Support

While Morph's sandbox.run_code() API supports multiple languages (Python, JavaScript, C++, Rust), the current implementation has only been used with Python. Additional work would be needed to fully support other languages in the Open-R1 codebase.

## Backward Compatibility

This implementation preserves backward compatibility with existing code:

1. E2B remains the default provider for code execution unless explicitly changed
2. Piston remains the default provider for IOI problems unless explicitly changed
3. No changes to function signatures that would break existing code

## PR Description for Upstream

**Title**: Add Morph Cloud as an alternative code execution provider

**Description**:

This PR introduces Morph Cloud as an alternative code execution provider for Open-R1, supporting two main use cases:

1. General code execution for GRPO training with Codeforces-style problems
2. IOI problem evaluation with multi-language support

The implementation maintains backward compatibility with existing E2B and Piston providers while offering Morph as an opt-in alternative through configuration options.