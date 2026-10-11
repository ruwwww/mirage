"""Graph builder for the transformer-block portion of Anima DiT.

The block layout intentionally follows Diffusers' ``CosmosTransformerBlock``
state-dict names.  Patchification, timestep embedding, and VAE transforms are
pipeline concerns; MPK receives flattened latent tokens, text context, and the
two timestep-condition tensors as runtime inputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from safetensors.torch import load_file

from ..graph_builder import GraphBuilder, MirageModelConfig
from ..utils import grid_for_rmsnorm_linear_layer
from ...model_registry import register_model_builder
from ....core import bfloat16


_BLOCK_KEY = re.compile(r"^(.*)transformer_blocks\.(\d+)\.norm1\.linear_1\.weight$")


@register_model_builder("anima", "Anima", "CosmosTransformer3DModel")
class AnimaBuilder(GraphBuilder):
    """Build Anima's AdaLN-Zero transformer stack as an MPK task graph.

    ``PersistentKernel`` predates Cosmos and exposes its legacy cached-attention
    ABI.  Anima therefore uses the explicit attention ABI below when available:
    ``cosmos_attention_layer`` or ``attention_layer`` accepting ``query``,
    ``key``, ``value``, ``q_norm``, ``k_norm``, and ``output``.  This keeps the
    builder independent of a particular attention kernel implementation while
    preserving the MPK layer boundary.
    """

    def __init__(self, mpk, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.eos_token_id = -1
        self.tokenizer = None
        self.model = None
        self._keep: list[torch.Tensor] = []
        self._buffers: dict[str, Any] = {}
        self._config_values: Mapping[str, Any] = {}
        self._weight_prefix = ""
        self._meta_tensors = getattr(mpk, "meta_tensors", {}) or {}

        # Anima is a fully persistent AOT graph.  If the host exposes scheduler
        # fields, make the model's requested execution policy explicit.
        for name in ("num_local_schedulers", "num_schedulers"):
            if hasattr(mpk, name):
                setattr(mpk, name, 0)

    # ------------------------------------------------------------------ config
    @staticmethod
    def _config_value(config: Any, name: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    def _set_architecture(self, state_dict: Mapping[str, torch.Tensor], config: Any = None,
                          num_layers: Optional[int] = None) -> None:
        block_ids = []
        prefix = None
        for key in state_dict:
            match = _BLOCK_KEY.match(key)
            if match:
                block_ids.append(int(match.group(2)))
                prefix = match.group(1)
        if not block_ids:
            raise ValueError("state_dict has no transformer_blocks.*.norm1.linear_1.weight")

        block_ids = sorted(set(block_ids))
        expected_ids = list(range(block_ids[-1] + 1))
        if block_ids != expected_ids:
            raise ValueError(f"transformer block ids must be contiguous, got {block_ids}")
        self._weight_prefix = prefix or ""

        first = block_ids[0]
        def shape(suffix: str) -> tuple[int, ...]:
            return tuple(self._weight(state_dict, first, suffix).shape)

        inferred_hidden = shape("attn1.to_q.weight")[0]
        inferred_head_dim = shape("attn1.norm_q.weight")[0]
        if inferred_head_dim <= 0 or inferred_hidden % inferred_head_dim:
            raise ValueError(
                f"attn1.norm_q.weight shape {shape('attn1.norm_q.weight')} "
                f"is incompatible with hidden size {inferred_hidden}"
            )
        inferred_intermediate = shape("ff.net.0.proj.weight")[0]
        inferred_adaln = shape("norm1.linear_1.weight")[0]
        inferred_text = shape("attn2.to_k.weight")[1]

        def choose(name: str, inferred: int) -> int:
            value = self._config_value(config, name, None)
            if value is None:
                return inferred
            value = int(value)
            if value != inferred:
                raise ValueError(f"{name}={value} disagrees with state_dict ({inferred})")
            return value

        self.hidden_size = choose("hidden_size", inferred_hidden)
        self.head_dim = choose("head_dim", inferred_head_dim)
        self.intermediate_size = choose("intermediate_size", inferred_intermediate)
        self.adaln_lora_dim = choose("adaln_lora_dim", inferred_adaln)
        self.text_embed_dim = choose("text_embed_dim", inferred_text)
        self.num_attention_heads = choose(
            "num_attention_heads",
            self.hidden_size // self.head_dim,
        )
        if self.hidden_size % self.head_dim:
            raise ValueError("hidden_size must be divisible by attention_head_dim")

        configured_layers = num_layers
        if configured_layers is None:
            configured_layers = self._config_value(config, "num_layers", None)
        self.num_layers = int(configured_layers or len(block_ids))
        if self.num_layers != len(block_ids):
            raise ValueError(
                f"num_layers={self.num_layers} but state_dict contains "
                f"{len(block_ids)} transformer blocks"
            )

        self.mlp_ratio = self.intermediate_size / self.hidden_size
        self._validate_weights(state_dict)

    def build_from_config(self, model_config: MirageModelConfig):
        state_dict = getattr(model_config, "state_dict", None)
        if not state_dict:
            raise ValueError("Anima requires model_config.state_dict")
        self._config_values = model_config
        return self.build_from_dict(
            state_dict,
            hidden_states=getattr(model_config, "hidden_states", None),
            encoder_hidden_states=getattr(model_config, "encoder_hidden_states", None),
            embedded_timestep=getattr(model_config, "embedded_timestep", None),
            temb=getattr(model_config, "temb", None),
            image_rotary_emb=getattr(model_config, "image_rotary_emb", None),
            attention_mask=getattr(model_config, "attention_mask", None),
            output=getattr(model_config, "output", None),
        )

    # ---------------------------------------------------------------- loading
    def build_from_model(
        self,
        model_name: str | None = None,
        model_path: str | None = None,
        model_path_or_name: str | None = None,
        **runtime_inputs,
    ):
        source = model_path_or_name or model_path or model_name
        if source is None:
            raise ValueError("model_path_or_name, model_name, or model_path is required")

        from diffusers import CosmosTransformer3DModel

        if isinstance(source, CosmosTransformer3DModel):
            model = source
            state_dict = model.state_dict()
        else:
            source_path = Path(source) if isinstance(source, str) else None
            if source_path is not None and source_path.is_file() and source_path.suffix == ".safetensors":
                model, state_dict = self._load_safetensors_model(source_path, CosmosTransformer3DModel)
            else:
                kwargs = {"torch_dtype": torch.bfloat16}
                try:
                    model = CosmosTransformer3DModel.from_pretrained(
                        str(source), subfolder="transformer", **kwargs
                    )
                except (OSError, ValueError):
                    model = CosmosTransformer3DModel.from_pretrained(str(source), **kwargs)
                state_dict = model.state_dict()

        self.model = model
        self._config_values = model.config
        self.model_name = model_name or model_path_or_name
        return self.build_from_dict(state_dict, **runtime_inputs)

    @staticmethod
    def _load_safetensors_model(path: Path, model_cls):
        config_candidates = (
            path.with_name("config.json"),
            path.parent / "transformer" / "config.json",
        )
        config_path = next((candidate for candidate in config_candidates if candidate.exists()), None)
        if config_path is None:
            raise ValueError(f"No Cosmos config.json found next to {path}")
        with config_path.open() as handle:
            config = json.load(handle)
        model = model_cls.from_config(config)
        state_dict = load_file(str(path), device="cpu")
        _, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(
                f"safetensors checkpoint does not match Cosmos model: "
                f"unexpected={unexpected}"
            )
        return model, state_dict

    # ------------------------------------------------------------- validation
    def _weight_key(self, layer: int, suffix: str) -> str:
        return f"{self._weight_prefix}transformer_blocks.{layer}.{suffix}"

    def _weight(self, state_dict: Mapping[str, torch.Tensor], layer: int, suffix: str):
        key = self._weight_key(layer, suffix)
        if key not in state_dict:
            raise ValueError(f"missing weight: {key}")
        return state_dict[key]

    def _validate_shape(self, state_dict, layer: int, suffix: str, expected: tuple[int, ...]):
        tensor = self._weight(state_dict, layer, suffix)
        actual = tuple(tensor.shape)
        if actual != expected:
            raise ValueError(
                f"{self._weight_key(layer, suffix)} has shape {actual}; expected {expected}"
            )
        return tensor

    def _validate_weights(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        h, d, adaln_rank, intermediate_size, text = (
            self.hidden_size,
            self.head_dim,
            self.adaln_lora_dim,
            self.intermediate_size,
            self.text_embed_dim,
        )
        for layer in range(self.num_layers):
            for norm in ("norm1", "norm2", "norm3"):
                self._validate_shape(
                    state_dict, layer, f"{norm}.linear_1.weight", (adaln_rank, h)
                )
                self._validate_shape(
                    state_dict, layer, f"{norm}.linear_2.weight", (3 * h, adaln_rank)
                )
            for attn, context_dim in (("attn1", h), ("attn2", text)):
                self._validate_shape(state_dict, layer, f"{attn}.norm_q.weight", (d,))
                self._validate_shape(state_dict, layer, f"{attn}.norm_k.weight", (d,))
                self._validate_shape(state_dict, layer, f"{attn}.to_q.weight", (h, h))
                self._validate_shape(state_dict, layer, f"{attn}.to_k.weight", (h, context_dim))
                self._validate_shape(state_dict, layer, f"{attn}.to_v.weight", (h, context_dim))
                self._validate_shape(state_dict, layer, f"{attn}.to_out.0.weight", (h, h))
            self._validate_shape(
                state_dict, layer, "ff.net.0.proj.weight", (intermediate_size, h)
            )
            self._validate_shape(
                state_dict, layer, "ff.net.2.weight", (h, intermediate_size)
            )

    # -------------------------------------------------------------- graph utils
    @staticmethod
    def _shape(tensor) -> tuple[int, ...]:
        if hasattr(tensor, "shape"):
            return tuple(tensor.shape)
        return tuple(tensor.dim(i) for i in range(tensor.num_dims))

    def _attach(self, tensor: torch.Tensor, name: str):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"weight {name} must be a torch.Tensor")
        tensor = tensor.contiguous()
        if torch.cuda.is_available() and not tensor.is_cuda:
            tensor = tensor.to(device="cuda", non_blocking=True)
        self._keep.append(tensor)
        return self.mpk.attach_input(torch_tensor=tensor, name=name)

    def _buffer(self, shape: tuple[int, ...], name: str):
        if name not in self._buffers:
            self._buffers[name] = self.mpk.new_tensor(
                dims=shape, dtype=bfloat16, name=name, io_category="cuda_tensor"
            )
        return self._buffers[name]

    def _runtime_tensor(self, explicit, names: tuple[str, ...], shape: tuple[int, ...]):
        tensor = explicit
        if tensor is None:
            for name in names:
                if name in self._meta_tensors:
                    tensor = self._meta_tensors[name]
                    break
        if tensor is None:
            return self._buffer(shape, names[0])
        if isinstance(tensor, torch.Tensor):
            return self._attach(tensor, names[0])
        return tensor

    def _grid(self, output_size: int) -> tuple[int, int, int]:
        grid = grid_for_rmsnorm_linear_layer(output_size)
        return (grid or 1, 1, 1)

    def _linear(self, input_tensor, weight, output_shape, name: str):
        output = self._buffer(output_shape, name)
        self.mpk.linear_layer(
            input=input_tensor,
            weight=weight,
            output=output,
            grid_dim=self._grid(output_shape[1]),
            block_dim=(128, 1, 1),
        )
        return output

    def _adaln_zero(self, state, embedded_timestep, temb, layer: int, norm_name: str,
                    rows: int, state_dict):
        h, adaln_rank = self.hidden_size, self.adaln_lora_dim
        normalized = self._buffer((rows, h), f"layer_{layer}_{norm_name}_rmsnorm")
        self.mpk.rmsnorm_layer(
            input=state,
            weight=self._attach(
                torch.ones(h, dtype=torch.bfloat16, device=self._weight(state_dict, layer, f"{norm_name}.linear_1.weight").device),
                f"layer_{layer}_{norm_name}_rmsnorm_weight",
            ),
            output=normalized,
            grid_dim=(rows, 1, 1),
            block_dim=(128, 1, 1),
        )
        linear_1 = self._attach(
            self._weight(state_dict, layer, f"{norm_name}.linear_1.weight"),
            f"layer_{layer}_{norm_name}_linear_1",
        )
        linear_2 = self._attach(
            self._weight(state_dict, layer, f"{norm_name}.linear_2.weight"),
            f"layer_{layer}_{norm_name}_linear_2",
        )
        modulation = self._linear(
            self.mpk.kn_graph.silu(
                self._linear(
                    embedded_timestep,
                    linear_1,
                    (rows, adaln_rank),
                    f"layer_{layer}_{norm_name}_down",
                )
            ),
            linear_2,
            (rows, 3 * h),
            f"layer_{layer}_{norm_name}_up",
        )
        if temb is not None:
            modulation = self.mpk.kn_graph.add(modulation, temb)
        shift, scale, gate = self.mpk.split(modulation, [h, h, h], 1)
        scaled = self.mpk.kn_graph.mul(normalized, scale)
        modulated = self.mpk.kn_graph.add(normalized, scaled)
        return self.mpk.kn_graph.add(modulated, shift), gate

    def _attention(self, query, key, value, q_norm, k_norm, output, *, layer: int,
                   kind: str, image_rotary_emb=None, attention_mask=None):
        attention = getattr(self.mpk, "cosmos_attention_layer", None)
        if attention is None:
            attention = getattr(self.mpk, "attention_layer", None)
        if attention is None:
            raise NotImplementedError(
                "Anima requires an MPK attention_layer accepting explicit "
                "query/key/value tensors"
            )
        try:
            attention(
                query=query,
                key=key,
                value=value,
                q_norm=q_norm,
                k_norm=k_norm,
                output=output,
                num_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                grid_dim=(self.num_attention_heads, 1, 1),
                block_dim=(128, 1, 1),
                is_causal=False,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )
        except TypeError as error:
            raise TypeError(
                f"MPK {kind}-attention layer must accept explicit "
                "query/key/value tensors"
            ) from error
        return output

    def _build_attention(self, state, context, layer: int, kind: str, rows: int,
                         context_rows: int, state_dict, image_rotary_emb=None,
                         attention_mask=None):
        h = self.hidden_size
        prefix = f"layer_{layer}_{kind}"
        q_weight = self._attach(
            self._weight(state_dict, layer, f"attn{1 if kind == 'self_attn' else 2}.to_q.weight"),
            f"{prefix}_to_q",
        )
        k_weight = self._attach(
            self._weight(state_dict, layer, f"attn{1 if kind == 'self_attn' else 2}.to_k.weight"),
            f"{prefix}_to_k",
        )
        v_weight = self._attach(
            self._weight(state_dict, layer, f"attn{1 if kind == 'self_attn' else 2}.to_v.weight"),
            f"{prefix}_to_v",
        )
        q = self._linear(state, q_weight, (rows, h), f"{prefix}_q")
        k = self._linear(context, k_weight, (context_rows, h), f"{prefix}_k")
        v = self._linear(context, v_weight, (context_rows, h), f"{prefix}_v")
        attn_name = "attn1" if kind == "self_attn" else "attn2"
        q_norm = self._attach(
            self._weight(state_dict, layer, f"{attn_name}.norm_q.weight"),
            f"{prefix}_q_norm",
        )
        k_norm = self._attach(
            self._weight(state_dict, layer, f"{attn_name}.norm_k.weight"),
            f"{prefix}_k_norm",
        )
        attention_out = self._buffer((rows, h), f"{prefix}_out")
        self._attention(
            q, k, v, q_norm, k_norm, attention_out, layer=layer, kind=kind,
            image_rotary_emb=image_rotary_emb, attention_mask=attention_mask,
        )
        out_weight = self._attach(
            self._weight(state_dict, layer, f"{attn_name}.to_out.0.weight"),
            f"{prefix}_to_out",
        )
        return self._linear(attention_out, out_weight, (rows, h), f"{prefix}_projected")

    # --------------------------------------------------------------- graph build
    def build_from_dict(self, state_dict: Mapping[str, torch.Tensor], with_lm_head: bool = False,
                        *, hidden_states=None, encoder_hidden_states=None,
                        embedded_timestep=None, temb=None, image_rotary_emb=None,
                        attention_mask=None, output=None, num_layers: Optional[int] = None):
        del with_lm_head
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping of weight names to tensors")
        self._set_architecture(state_dict, self._config_values, num_layers)
        rows = getattr(self.mpk, "max_num_batched_tokens", None) or 1
        hidden_states = self._runtime_tensor(hidden_states, ("hidden_states", "latents"),
                                             (rows, self.hidden_size))
        embedded_timestep = self._runtime_tensor(
            embedded_timestep, ("embedded_timestep",), (rows, self.hidden_size)
        )
        encoder_hidden_states = self._runtime_tensor(
            encoder_hidden_states, ("encoder_hidden_states", "text_context"),
            (rows, self.text_embed_dim)
        )
        if temb is None:
            temb = self._meta_tensors.get("temb")
        if isinstance(temb, torch.Tensor):
            temb = self._attach(temb, "temb")
        context_rows = self._shape(encoder_hidden_states)[0]
        state = hidden_states
        for layer in range(self.num_layers):
            normed, gate = self._adaln_zero(
                state, embedded_timestep, temb, layer, "norm1", rows, state_dict
            )
            attn = self._build_attention(
                normed, normed, layer, "self_attn", rows, rows, state_dict,
                image_rotary_emb, None,
            )
            state = self.mpk.kn_graph.add(
                state, self.mpk.kn_graph.mul(gate, attn)
            )

            normed, gate = self._adaln_zero(
                state, embedded_timestep, temb, layer, "norm2", rows, state_dict
            )
            attn = self._build_attention(
                normed, encoder_hidden_states, layer, "cross_attn", rows,
                context_rows, state_dict, None, attention_mask
            )
            state = self.mpk.kn_graph.add(
                state, self.mpk.kn_graph.mul(gate, attn)
            )

            normed, gate = self._adaln_zero(
                state, embedded_timestep, temb, layer, "norm3", rows, state_dict
            )
            ff_in = self._attach(
                self._weight(state_dict, layer, "ff.net.0.proj.weight"),
                f"layer_{layer}_ff_in",
            )
            ff_out = self._attach(
                self._weight(state_dict, layer, "ff.net.2.weight"),
                f"layer_{layer}_ff_out",
            )
            ff_hidden = self._linear(
                normed, ff_in, (rows, self.intermediate_size),
                f"layer_{layer}_ff_hidden",
            )
            ff_hidden = self.mpk.kn_graph.gelu(ff_hidden)
            ff_result = self._linear(
                ff_hidden, ff_out, (rows, self.hidden_size),
                f"layer_{layer}_ff_result",
            )
            state = self.mpk.kn_graph.add(
                state, self.mpk.kn_graph.mul(gate, ff_result)
            )

        if output is not None:
            output = self._runtime_tensor(output, ("output",), (rows, self.hidden_size))
            if not hasattr(self.mpk, "identity_layer"):
                raise NotImplementedError(
                    "Anima output binding requires MPK identity_layer"
                )
            self.mpk.identity_layer(
                input=state,
                output=output,
                grid_dim=(rows, 1, 1),
                block_dim=(128, 1, 1),
            )
            state = output
        self.x = self.returned_hidden_state = state
        return state

    def build_euler_update(self, latents, velocity, delta_t, output=None):
        """Append one Euler flow-matching update to the persistent graph.

        A caller can unroll this method for a fixed sampling horizon when
        constructing the AOT graph; the graph itself contains no scheduler.
        """
        if isinstance(delta_t, (float, int)):
            shape = self._shape(latents)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            delta_t = self._attach(
                torch.full(shape, float(delta_t), dtype=torch.bfloat16, device=device),
                "anima_delta_t",
            )
        updated = self.mpk.kn_graph.add(
            latents, self.mpk.kn_graph.mul(velocity, delta_t)
        )
        if output is not None and hasattr(self.mpk, "identity_layer"):
            self.mpk.identity_layer(
                input=updated,
                output=output,
                grid_dim=(self._shape(updated)[0], 1, 1),
                block_dim=(128, 1, 1),
            )
            return output
        return updated
