import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
PYTHON_ROOT = ROOT / "python"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_anima_builder():
    """Load the builder without requiring Mirage's optional native extension."""
    packages = {
        "mirage": PYTHON_ROOT / "mirage",
        "mirage.mpk": PYTHON_ROOT / "mirage" / "mpk",
        "mirage.mpk.models": PYTHON_ROOT / "mirage" / "mpk" / "models",
        "mirage.mpk.models.anima": PYTHON_ROOT / "mirage" / "mpk" / "models" / "anima",
    }
    for name, path in packages.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    core = types.ModuleType("mirage.core")
    core.bfloat16 = "bf16"
    sys.modules["mirage.core"] = core

    persistent_kernel = types.ModuleType("mirage.mpk.persistent_kernel")
    persistent_kernel.PersistentKernel = object
    sys.modules["mirage.mpk.persistent_kernel"] = persistent_kernel

    _load_module(
        "mirage.mpk.model_registry",
        PYTHON_ROOT / "mirage" / "mpk" / "model_registry.py",
    )
    _load_module(
        "mirage.mpk.models.graph_builder",
        PYTHON_ROOT / "mirage" / "mpk" / "models" / "graph_builder.py",
    )
    _load_module(
        "mirage.mpk.models.utils",
        PYTHON_ROOT / "mirage" / "mpk" / "models" / "utils.py",
    )
    return _load_module(
        "mirage.mpk.models.anima.builder",
        PYTHON_ROOT / "mirage" / "mpk" / "models" / "anima" / "builder.py",
    )


@dataclass
class FakeTensor:
    shape: tuple
    name: str = ""

    @property
    def num_dims(self):
        return len(self.shape)

    def dim(self, index):
        return self.shape[index]


class FakeGraph:
    def silu(self, tensor):
        return FakeTensor(tensor.shape, f"silu({tensor.name})")

    def gelu(self, tensor):
        return FakeTensor(tensor.shape, f"gelu({tensor.name})")

    def add(self, lhs, rhs):
        assert lhs.shape == rhs.shape
        return FakeTensor(lhs.shape, f"({lhs.name}+{rhs.name})")

    def mul(self, lhs, rhs):
        assert lhs.shape == rhs.shape
        return FakeTensor(lhs.shape, f"({lhs.name}*{rhs.name})")


class FakeMPK:
    def __init__(self, rows=4):
        self.max_num_batched_tokens = rows
        self.world_size = 1
        self.mpi_rank = 0
        self.num_local_schedulers = 4
        self.num_schedulers = 4
        self.meta_tensors = {}
        self.kn_graph = FakeGraph()
        self.attached = {}
        self.layers = []

    def attach_input(self, torch_tensor, name):
        tensor = FakeTensor(tuple(torch_tensor.shape), name)
        self.attached[name] = torch_tensor
        return tensor

    def new_tensor(self, dims, dtype, name, io_category="cuda_tensor"):
        return FakeTensor(tuple(dims), name)

    def view(self, tensor, new_shape):
        assert torch.tensor(tensor.shape).prod().item() == torch.tensor(new_shape).prod().item()
        return FakeTensor(tuple(new_shape), f"view({tensor.name})")

    def split(self, tensor, sizes, dim):
        assert dim == 1
        start = 0
        result = []
        for size in sizes:
            shape = list(tensor.shape)
            shape[dim] = size
            result.append(FakeTensor(tuple(shape), f"split({tensor.name},{start})"))
            start += size
        return result

    def linear_layer(self, input, weight, output, grid_dim, block_dim, bias=None):
        assert input.num_dims == output.num_dims == 2
        assert input.dim(1) == weight.dim(1)
        assert output.dim(1) == weight.dim(0)
        self.layers.append(("linear", output.name, tuple(weight.shape)))

    def rmsnorm_layer(self, input, weight, output, grid_dim, block_dim):
        assert input.num_dims == output.num_dims == 2
        assert input.shape == output.shape
        assert weight.num_dims == 1 and weight.dim(0) == input.dim(1)
        self.layers.append(("rmsnorm", output.name, tuple(weight.shape)))

    def attention_layer(self, **kwargs):
        query = kwargs["query"]
        key = kwargs["key"]
        value = kwargs["value"]
        output = kwargs["output"]
        assert kwargs["q_norm"].shape == (4,)
        assert kwargs["k_norm"].shape == (4,)
        assert query.shape == key.shape == value.shape == output.shape
        self.layers.append(("attention", output.name, kwargs["num_heads"]))


def _miniature_model():
    from diffusers import CosmosTransformer3DModel

    return CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=2,
        text_embed_dim=6,
        adaln_lora_dim=3,
        max_size=(4, 8, 8),
        extra_pos_embed_type=None,
    )


def test_anima_builder_is_registered_under_all_requested_names():
    builder_module = _load_anima_builder()
    registry = sys.modules["mirage.mpk.model_registry"]

    for name in ("anima", "Anima", "CosmosTransformer3DModel"):
        assert registry.get_builder(name) is builder_module.AnimaBuilder


def test_anima_builder_maps_a_two_block_cosmos_model_to_mpk_layers():
    builder_module = _load_anima_builder()
    model = _miniature_model()
    mpk = FakeMPK(rows=4)
    builder = builder_module.AnimaBuilder(mpk)

    builder.build_from_dict(
        model.state_dict(),
        hidden_states=FakeTensor((4, 8), "hidden_states"),
        encoder_hidden_states=FakeTensor((4, 6), "encoder_hidden_states"),
        embedded_timestep=FakeTensor((4, 8), "embedded_timestep"),
        temb=FakeTensor((4, 24), "temb"),
    )

    assert mpk.num_local_schedulers == 0
    assert mpk.num_schedulers == 0
    assert len([layer for layer in mpk.layers if layer[0] == "attention"]) == 4
    assert len([layer for layer in mpk.layers if layer[0] == "rmsnorm"]) == 6
    assert len([layer for layer in mpk.layers if layer[0] == "linear"]) == 32
    assert len(mpk.attached) >= 40
    assert builder.num_layers == 2
    assert builder.hidden_size == 8
    assert builder.text_embed_dim == 6


def test_anima_builder_accepts_mirage_model_config():
    builder_module = _load_anima_builder()
    graph_builder = sys.modules["mirage.mpk.models.graph_builder"]
    model = _miniature_model()
    config = graph_builder.MirageModelConfig(
        hidden_size=8,
        head_dim=4,
        num_layers=2,
        state_dict=model.state_dict(),
        with_lm_head=False,
    )

    builder = builder_module.AnimaBuilder(FakeMPK(rows=4))
    builder.build_from_config(config)

    assert builder.num_layers == 2
    assert builder.intermediate_size == 32
    assert builder.adaln_lora_dim == 3


def test_anima_builder_loads_a_diffusers_model_instance():
    builder_module = _load_anima_builder()
    model = _miniature_model()
    builder = builder_module.AnimaBuilder(FakeMPK(rows=4))

    builder.build_from_model(
        model,
        hidden_states=FakeTensor((4, 8), "hidden_states"),
        encoder_hidden_states=FakeTensor((4, 6), "encoder_hidden_states"),
        embedded_timestep=FakeTensor((4, 8), "embedded_timestep"),
    )

    assert builder.model is model
    assert builder.num_layers == 2


def test_anima_builder_can_append_an_euler_flow_matching_update():
    builder_module = _load_anima_builder()
    builder = builder_module.AnimaBuilder(FakeMPK(rows=4))
    latents = FakeTensor((4, 8), "latents")
    velocity = FakeTensor((4, 8), "velocity")

    updated = builder.build_euler_update(latents, velocity, 0.25)

    assert updated.shape == (4, 8)


def test_anima_builder_rejects_a_malformed_weight_shape():
    builder_module = _load_anima_builder()
    model = _miniature_model()
    state_dict = dict(model.state_dict())
    state_dict["transformer_blocks.0.attn1.norm_q.weight"] = torch.empty(3)

    with pytest.raises(ValueError, match="attn1.norm_q.weight"):
        builder_module.AnimaBuilder(FakeMPK()).build_from_dict(state_dict)
