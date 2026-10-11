# Implementation Plan: Anima DiT Support in Mirage MPK

## Phase 1: Model Builder Implementation
- [ ] Create `python/mirage/mpk/models/anima/__init__.py` exporting `AnimaBuilder`.
- [ ] Create `python/mirage/mpk/models/anima/builder.py`:
  - Register with `@register_model_builder("anima", "Anima", "CosmosTransformer3DModel")`.
  - Implement `AnimaBuilder(GraphBuilder)` supporting:
    - `build_from_config(model_config: MirageModelConfig)`
    - `build_from_model(model_path_or_name: str)` supporting both HuggingFace / Diffusers `CosmosTransformer3DModel` and safetensors checkpoint loading.
    - `build_from_dict(state_dict: dict, ...)` mapping all 28 blocks of `CosmosTransformerBlock` (norm1, attn1, norm2, attn2, norm3, ff) to Mirage MPK layer calls (`mpk.linear_layer`, `mpk.rmsnorm_layer`, `mpk.attention_layer`, etc.).
- [ ] Register `anima` in `python/mirage/mpk/models/__init__.py`.

## Phase 2: Verification Harness & Unit Tests
- [ ] Create `tests/test_anima_builder.py`:
  - Test builder registration retrieval via `get_builder("anima")`.
  - Test config mapping with miniature/2-layer `CosmosTransformer3DModel`.
  - Verify tensor attachment, weight shape validation, and graph construction.
- [ ] Run test suite with `/home/kuroko/.conda/envs/ai/bin/python -m pytest tests/test_anima_builder.py -v`.

## Phase 3: Git Commit & Delivery
- [ ] Ensure clean code formatting and passing tests.
- [ ] Commit all changes to branch `feature/anima-dit`.
