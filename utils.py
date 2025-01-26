# coding=utf-8
#
# Copyright 2025 Michael Eberhard
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import List, Tuple

import torch
from accelerate import init_empty_weights
from transformers.models.qwen2.modeling_qwen2 import (Qwen2DecoderLayer,
                                                      Qwen2ForCausalLM)

from modeling.modeling_qwen_stages import (Qwen2ModelEnd, Qwen2ModelMiddle,
                                           Qwen2ModelStart)
from modeling.stage_wrappers import EndStage, MiddleStage, StartStage


def _split_model(
    qwen: Qwen2ForCausalLM, n_devices: int, retain_old_weights: bool = False
) -> Tuple[Qwen2ModelStart, List[Qwen2ModelMiddle], Qwen2ModelEnd]:
    """
    Split the Qwen2ForCausalLM into Qwen2ModelStart, Qwen2ModelMiddle, and Qwen2ModelEnd.

    Args:
        qwen: The Qwen2 model to split.
        qwen2_config: The configuration of the Qwen2 model.
        n_devices: The number of devices to split the model into.
        retain_old_weights: Wether to keep the pretrained weights of the model.

    Returns:
        A tuple containing the start, middle, and end stages of the model.
    """
    assert (
        n_devices > 1
    ), "n_devices must be greater than 1. Currently only multi-GPU training is supported."
    qwen2_config = qwen.config
    n_layers = qwen2_config.num_hidden_layers
    layers_per_device = n_layers // n_devices

    start = Qwen2ModelStart(qwen2_config)
    middles = [
        Qwen2ModelMiddle(
            qwen2_config, layers_per_device, start_idx=i * layers_per_device
        )
        for i in range(n_devices)
    ]
    end = Qwen2ModelEnd(qwen2_config)

    start.embed_tokens.load_state_dict(qwen.model.embed_tokens.state_dict())
    start.rotary_emb.load_state_dict(qwen.model.rotary_emb.state_dict())

    if not retain_old_weights:
        del qwen.model.embed_tokens
        del qwen.model.rotary_emb

    for i, middle in enumerate(middles):
        for j in range(layers_per_device):
            layer = qwen.model.layers[i * layers_per_device + j]
            middle.layers[j].load_state_dict(layer.state_dict())
            if not retain_old_weights:
                del layer.self_attn
                del layer.mlp
                del layer.input_layernorm
                del layer.post_attention_layernorm

    end.lm_head.load_state_dict(qwen.lm_head.state_dict())
    end.norm.load_state_dict(qwen.model.norm.state_dict())

    if not retain_old_weights:
        del qwen.lm_head
        del qwen.model.norm

    return start, middles, end


def _stage_model(
    start: Qwen2ModelStart,
    middles: List[Qwen2ModelMiddle],
    end: Qwen2ModelEnd,
    n_devices: int,
) -> Tuple[StartStage, List[MiddleStage], EndStage]:
    assert n_devices >= 2, "Need at least 2 devices"
    assert (
        len(middles) == n_devices
    ), f"Number of middle blocks ({len(middles)}) must match number of devices ({n_devices})"

    if n_devices == 2:
        start_stage = StartStage(start, middles[0])
        middle_stages = []
        end_stage = EndStage(middles[1], end)
    else:
        start_stage = StartStage(start, middles[0])
        middle_stages = [MiddleStage(middle) for middle in middles[1:-1]]
        end_stage = EndStage(middles[-1], end)

    return start_stage, middle_stages, end_stage


def split_models_to_stages(
    qwen: Qwen2ForCausalLM, n_devices: int, retain_old_weights: bool = False
) -> Tuple[StartStage, List[MiddleStage], EndStage]:
    """
    Split the Qwen2ForCausalLM into stages.

    Args:
        qwen: The Qwen2 model to split.
        qwen2_config: The configuration of the Qwen2 model.
        n_devices: The number of devices to split the model into.
        retain_old_weights: Wether to keep the pretrained weights of the model.

    Returns:
        A tuple containing the start, middle, and end stages of the model.
    """
    start, middles, end = _split_model(qwen, n_devices, retain_old_weights)
    return _stage_model(start, middles, end, n_devices)


def stages_to_device(
    start_stage: StartStage,
    middle_stages: List[MiddleStage],
    end_stage: EndStage,
    devices: List[torch.device],
    loss_device: torch.device | None = None,
) -> Tuple[StartStage, List[MiddleStage], EndStage]:
    start_stage.to(devices[0])
    start_stage.set_target_device(devices[0])
    for i, middle_stage in enumerate(middle_stages):
        middle_stage.to(devices[i + 1])
        middle_stage.set_target_device(devices[i + 1])
    end_stage.to(devices[-1])
    end_stage.set_target_device(devices[-1])
    end_stage.set_loss_device(loss_device)
    return start_stage, middle_stages, end_stage


def get_optimizers(
    start_stage: StartStage,
    middle_stages: List[MiddleStage],
    end_stage: EndStage,
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
) -> Tuple[torch.optim.Optimizer, List[torch.optim.Optimizer]]:
    start_optimizer = optimizer_class(
        start_stage.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    middle_optimizers = [
        optimizer_class(
            middle_stage.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        for middle_stage in middle_stages
    ]
    end_optimizer = optimizer_class(
        end_stage.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    return start_optimizer, middle_optimizers, end_optimizer


def stages_to_model_shallow(
    start_stage: StartStage,
    middle_stages: List[MiddleStage],
    end_stage: EndStage,
) -> Qwen2ForCausalLM:
    config = start_stage.start.config
    with init_empty_weights():
        qwen = Qwen2ForCausalLM(config)

    qwen.model.rotary_emb = start_stage.start.rotary_emb
    qwen.model.embed_tokens = start_stage.start.embed_tokens

    layers_per_stage = len(start_stage.middle.layers)

    for j in range(layers_per_stage):
        qwen.model.layers[j] = start_stage.middle.layers[j]

    for i, middle_stage in enumerate(middle_stages):
        for j in range(layers_per_stage):
            layer = middle_stage.middle.layers[j]
            qwen.model.layers[(i + 1) * layers_per_stage + j] = layer

    start_idx = (len(middle_stages) + 1) * layers_per_stage
    for j in range(layers_per_stage):
        qwen.model.layers[start_idx + j] = end_stage.middle.layers[j]

    qwen.model.norm = end_stage.end.norm
    qwen.lm_head = end_stage.end.lm_head

    return qwen


def stages_to_model(
    start_stage: StartStage, middle_stages: List[MiddleStage], end_stage: EndStage
) -> Qwen2ForCausalLM:
    """
    Combine the split stages back into a Qwen2ForCausalLM model.

    Args:
        start_stage: The start stage containing embedding layers
        middle_stages: List of middle stages containing transformer layers
        end_stage: The end stage containing the LM head and final norm

    Returns:
        A Qwen2ForCausalLM model with weights loaded from the stages
    """
    qwen = Qwen2ForCausalLM(start_stage.start.config)

    # something is wrong here:
    qwen.model.embed_tokens.load_state_dict(start_stage.start.embed_tokens.state_dict())
    qwen.model.rotary_emb.load_state_dict(start_stage.start.rotary_emb.state_dict())

    layers_per_stage = len(start_stage.middle.layers)

    for j in range(layers_per_stage):
        layer = start_stage.middle.layers[j]
        target_layer = qwen.model.layers[j]
        target_layer.load_state_dict(layer.state_dict())

    for i, middle_stage in enumerate(middle_stages):
        for j in range(layers_per_stage):
            layer = middle_stage.middle.layers[j]
            target_layer = qwen.model.layers[(i + 1) * layers_per_stage + j]
            target_layer.load_state_dict(layer.state_dict())

    start_idx = (len(middle_stages) + 1) * layers_per_stage
    for j in range(layers_per_stage):
        layer = end_stage.middle.layers[j]
        target_layer = qwen.model.layers[start_idx + j]
        target_layer.load_state_dict(layer.state_dict())

    qwen.model.norm.load_state_dict(end_stage.end.norm.state_dict())
    qwen.lm_head.load_state_dict(end_stage.end.lm_head.state_dict())

    return qwen


def copy_layer_weights(
    source_layer: Qwen2DecoderLayer, target_layer: Qwen2DecoderLayer
):
    target_layer.self_attn.q_proj.load_state_dict(
        source_layer.self_attn.q_proj.state_dict()
    )
    target_layer.self_attn.k_proj.load_state_dict(
        source_layer.self_attn.k_proj.state_dict()
    )
    target_layer.self_attn.v_proj.load_state_dict(
        source_layer.self_attn.v_proj.state_dict()
    )
    target_layer.self_attn.o_proj.load_state_dict(
        source_layer.self_attn.o_proj.state_dict()
    )
    target_layer.mlp.gate_proj.load_state_dict(source_layer.mlp.gate_proj.state_dict())
    target_layer.mlp.up_proj.load_state_dict(source_layer.mlp.up_proj.state_dict())
    target_layer.mlp.down_proj.load_state_dict(source_layer.mlp.down_proj.state_dict())
    target_layer.input_layernorm.load_state_dict(
        source_layer.input_layernorm.state_dict()
    )
    target_layer.post_attention_layernorm.load_state_dict(
        source_layer.post_attention_layernorm.state_dict()
    )
