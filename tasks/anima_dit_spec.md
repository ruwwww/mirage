# Anima DiT Architecture Specification for Mirage MPK

## Target Model Overview
- **Model:** Anima DiT (Circlestone Labs / Anima-Base-v1.0 / Anima-Turbo)
- **Reference Implementation:** `diffusers.models.transformers.transformer_cosmos.CosmosTransformer3DModel` (stock Diffusers >= 0.39.0)
- **Target Role:** Diffusion Transformer backbone for video/image generation, compiled as a persistent AOT task graph in Mirage MPK.

## Architectural Dimensions & Hyperparameters
- `hidden_size`: 2048 (`num_attention_heads * attention_head_dim`)
- `num_attention_heads`: 16
- `attention_head_dim`: 128
- `num_layers`: 28 (configurable, default 28 in full model)
- `mlp_ratio`: 4.0 (`ffn_hidden_dim` = 8192)
- `adaln_lora_dim`: 256 (low-rank bottleneck for modulation)
- `text_embed_dim`: 1024 (T5 / text condition channels for Cross-Attention)
- `in_channels`: 16, `out_channels`: 16
- `patch_size`: (1, 2, 2)
- `qk_norm`: "rms_norm" (RMSNorm applied to Q and K with head_dim=128)

## State Dict Weight Hierarchy & Mapping
For each layer $i \in [0, \text{num\_layers}-1]$:
```
transformer_blocks.{i}.norm1.linear_1.weight       : [256, 2048]
transformer_blocks.{i}.norm1.linear_2.weight       : [6144, 256]   # 3 x 2048 (shift, scale, gate for self-attn)
transformer_blocks.{i}.attn1.norm_q.weight         : [128]
transformer_blocks.{i}.attn1.norm_k.weight         : [128]
transformer_blocks.{i}.attn1.to_q.weight           : [2048, 2048]
transformer_blocks.{i}.attn1.to_k.weight           : [2048, 2048]
transformer_blocks.{i}.attn1.to_v.weight           : [2048, 2048]
transformer_blocks.{i}.attn1.to_out.0.weight       : [2048, 2048]

transformer_blocks.{i}.norm2.linear_1.weight       : [256, 2048]
transformer_blocks.{i}.norm2.linear_2.weight       : [6144, 256]   # 3 x 2048 (shift, scale, gate for cross-attn)
transformer_blocks.{i}.attn2.norm_q.weight         : [128]
transformer_blocks.{i}.attn2.norm_k.weight         : [128]
transformer_blocks.{i}.attn2.to_q.weight           : [2048, 2048]
transformer_blocks.{i}.attn2.to_k.weight           : [2048, 1024]  # from text_embed_dim
transformer_blocks.{i}.attn2.to_v.weight           : [2048, 1024]  # from text_embed_dim
transformer_blocks.{i}.attn2.to_out.0.weight       : [2048, 2048]

transformer_blocks.{i}.norm3.linear_1.weight       : [256, 2048]
transformer_blocks.{i}.norm3.linear_2.weight       : [6144, 256]   # 3 x 2048 (shift, scale, gate for ffn)
transformer_blocks.{i}.ff.net.0.proj.weight        : [8192, 2048]
transformer_blocks.{i}.ff.net.2.weight             : [2048, 8192]
```

## Mirage Task Graph Construction Requirements
1. **Persistent Execution Mode:**
   - 100% Ahead-Of-Time (AOT) task graph.
   - Schedulers = 0 (all SMs allocated to worker CTAs).
2. **Layer Sub-Tasks:**
   - AdaLN-Zero modulation: Down-projection `linear_1` followed by up-projection `linear_2`, producing `shift, scale, gate`.
   - Modulated RMSNorm: $x_{\text{norm}} = \text{RMSNorm}(x) \cdot (1 + \text{scale}) + \text{shift}$.
   - Self-Attention: $Q, K, V$ GEMM with QK-RMSNorm, followed by output projection GEMM.
   - Cross-Attention: $Q$ from hidden states, $K, V$ from context embeddings ($D_{\text{text}}=1024$), output projection GEMM.
   - FeedForward: GEMM $W_1$ (2048 -> 8192), GeLU activation, GEMM $W_2$ (8192 -> 2048).
   - Gated Residual Accumulation: $x \leftarrow x + \text{gate} \cdot \text{sublayer\_out}$.
3. **Multi-Step Euler Flow Matching In-Kernel Loop:**
   - Support terminal ODE update task: $x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta$.
   - Iteration counter decrements until sampling horizon (e.g. 25 or 50 steps) completes in a single persistent kernel launch.
