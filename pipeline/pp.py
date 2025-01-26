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

from queue import Queue
from threading import Thread
from typing import List

import torch

from modeling.stage_wrappers import EndStage, MiddleStage, StartStage
from utils import stages_to_model_shallow


class Pipeline:
    def __init__(
        self,
        start_stage: StartStage,
        middle_stages: List[MiddleStage],
        end_stage: EndStage,
        microbatch_size: int,
    ):
        self.start_stage = start_stage
        self.middle_stages = middle_stages
        self.end_stage = end_stage
        self.microbatch_size = microbatch_size
        self._init_queues()
        self._start_threads()

    def _init_queues(self):
        self.start_queue = Queue()
        self.middle_queues = [Queue() for _ in range(len(self.middle_stages) + 1)]
        self.end_queue = Queue()
        self.ready_queue = Queue()

    def _start_threads(self):
        self.start_thread = Thread(
            target=start_worker,
            args=(
                self.start_stage,
                self.start_queue,
                self.middle_queues[0],
                self.ready_queue,
            ),
        )
        self.middle_threads = [
            Thread(
                target=middle_worker,
                args=(
                    self.middle_stages[i],
                    self.middle_queues[i],
                    self.middle_queues[i + 1],
                    self.ready_queue,
                ),
            )
            for i in range(len(self.middle_stages))
        ]
        self.end_thread = Thread(
            target=end_worker,
            args=(
                self.end_stage,
                self.middle_queues[-1],
                self.end_queue,
                self.ready_queue,
            ),
        )
        self.start_thread.start()
        for thread in self.middle_threads:
            thread.start()
        self.end_thread.start()
        self._block_until_ready()

    def _block_until_ready(self):
        for _ in range(len(self.middle_stages) + 2):
            self.ready_queue.get()

    def _shutdown_threads(self):
        self.start_queue.put(None)
        self.end_queue.put(None)
        for queue in self.middle_queues:
            queue.put(None)
        self.start_thread.join()
        for thread in self.middle_threads:
            thread.join()
        self.end_thread.join()

    def _get_results(self):
        results = []
        while not self.end_queue.empty():
            item = self.end_queue.get()
            if item is None:
                continue
            results.append(item)
        return results

    def process_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Processes a batch of input tensors. Blocks until all microbatches are processed.

        Args:
            input_tensor: shape [batch, seq_len], the input tensor to process.
        """
        # print(
        #    f"Processing batch of size {input_ids.shape[0]} with microbatch size {self.microbatch_size}"
        # )
        self.to_target()
        for i in range(0, input_ids.shape[0], self.microbatch_size):
            microbatch_ids = input_ids[i : i + self.microbatch_size]
            microbatch_attention_mask = attention_mask[i : i + self.microbatch_size]
            self.start_queue.put((microbatch_ids, microbatch_attention_mask))
        self.start_queue.join()
        for queue in self.middle_queues:
            queue.join()
        results = self._get_results()
        results = torch.cat(results, dim=0)
        return results

    def to_target(self):
        if self.start_stage.device != self.start_stage.target_device:
            self.start_stage = self.start_stage.to(self.start_stage.target_device)
        for middle_stage in self.middle_stages:
            if middle_stage.device != middle_stage.target_device:
                middle_stage = middle_stage.to(middle_stage.target_device)
        if self.end_stage.device != self.end_stage.target_device:
            self.end_stage = self.end_stage.to(self.end_stage.target_device)

    def save_full_model(self, path: str):
        """
        Combine the split stages back into a Qwen2ForCausalLM model.

        Args:
            start_stage: The start stage containing embedding layers
            middle_stages: List of middle stages containing transformer layers
            end_stage: The end stage containing the LM head and final norm

        Returns:
            A Qwen2ForCausalLM model with weights loaded from the stages
        """
        model = self.copy_shallow_model()
        model.save_pretrained(path)

    def copy_shallow_model(self):
        return stages_to_model_shallow(
            self.start_stage, self.middle_stages, self.end_stage
        )


def move_to(hs, cm, pe, pi, cp, device):
    hs = hs.to(device)
    cm = cm.to(device) if cm is not None else None
    pe = [p.to(device) for p in pe]
    pi = pi.to(device) if pi is not None else None
    cp = cp.to(device) if cp is not None else None
    return hs, cm, pe, pi, cp


def start_worker(
    start_stage: StartStage, in_queue: Queue, out_queue: Queue, ready_queue: Queue
):
    ready_queue.put(True)
    while True:
        input_ids, attention_mask = in_queue.get()
        if input_ids is None:
            break
        input_ids = input_ids.to(start_stage.device)
        attention_mask = attention_mask.to(start_stage.device)
        hs, cm, pe, pi, cp = start_stage(input_ids, attention_mask)
        out_queue.put((hs, cm, pe, pi, cp))
        in_queue.task_done()


def middle_worker(
    middle_stage: MiddleStage, in_queue: Queue, out_queue: Queue, ready_queue: Queue
):
    ready_queue.put(True)
    while True:
        item = in_queue.get()
        if item is None:
            break
        hs, cm, pe, pi, cp = item
        hs, cm, pe, pi, cp = move_to(hs, cm, pe, pi, cp, middle_stage.device)
        hs, cm, pe, pi, cp = middle_stage(hs, cm, pe, pi, cp)
        out_queue.put((hs, cm, pe, pi, cp))
        in_queue.task_done()


def end_worker(
    end_stage: EndStage, in_queue: Queue, out_queue: Queue, ready_queue: Queue
):
    ready_queue.put(True)
    while True:
        item = in_queue.get()
        if item is None:
            break
        hs, cm, pe, pi, cp = item
        hs, cm, pe, pi, cp = move_to(hs, cm, pe, pi, cp, end_stage.device)
        logits = end_stage(hs, cm, pe, pi, cp)
        logits = logits.to(end_stage.loss_device)
        out_queue.put(logits)
        in_queue.task_done()
