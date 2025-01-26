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


from typing import List

import torch
import torch.nn as nn

from modeling.modeling_qwen_stages import (Qwen2ModelEnd, Qwen2ModelMiddle,
                                           Qwen2ModelStart)


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_device: torch.device | None = None

    def set_target_device(self, device: str):
        self.target_device = torch.device(device)

    @property
    def device(self):
        return next(self.parameters()).device


class StartStage(Stage):
    def __init__(self, start: Qwen2ModelStart, middle: Qwen2ModelMiddle):
        super().__init__()
        self.start = start
        self.middle = middle

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hs, cm, pe, pi, cp = self.start(input_ids, attention_mask)
        hs, cm, pe, pi, cp = self.middle(hs, cm, pe, pi, cp)
        return hs, cm, pe, pi, cp


class MiddleStage(Stage):
    def __init__(self, middle: Qwen2ModelMiddle):
        super().__init__()
        self.middle = middle

    def forward(
        self,
        hs: torch.Tensor,
        cm: torch.Tensor,
        pe: List[torch.Tensor],
        pi: torch.Tensor,
        cp: torch.Tensor,
    ):
        hs, cm, pe, pi, cp = self.middle(hs, cm, pe, pi, cp)
        return hs, cm, pe, pi, cp


class EndStage(Stage):
    def __init__(self, middle: Qwen2ModelMiddle, end: Qwen2ModelEnd):
        super().__init__()
        self.middle = middle
        self.end = end
        self.loss_device: torch.device | None = None

    def forward(
        self,
        hs: torch.Tensor,
        cm: torch.Tensor,
        pe: List[torch.Tensor],
        pi: torch.Tensor,
        cp: torch.Tensor,
    ):
        hs, cm, pe, pi, cp = self.middle.forward(hs, cm, pe, pi, cp)
        return self.end(hs, num_logits_to_keep=0)

    def set_loss_device(self, device: torch.device):
        self.loss_device = device
