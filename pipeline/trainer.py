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

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

import wandb
from pipeline.grpo import grpo_loss
from pipeline.pp import Pipeline
from pipeline.reward import get_rewards
from utils import get_optimizers, split_models_to_stages, stages_to_device


class GRPOTrainer:
    def __init__(
        self,
        hf_model_name: str,
        learning_rate: float,
        weight_decay: float,
        devices: List[str],
        batch_size: int,
        microbatch_size: int,
        project_dir: str,
        system_prompt: str | None = None,
        loss_device: str | None = None,
        verbose: bool = False,
        use_wandb: bool = True,
    ):
        self.hf_model_name = hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, padding_side="left"
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.devices = devices
        if loss_device is None:
            self.loss_device = self.devices[1]
        else:
            self.loss_device = loss_device
        self.idle_device = "cpu"
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        self.project_dir = project_dir
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else "You are a helpful assistant."
        )
        self.verbose = verbose
        pipe = self._init_pipeline()
        self.pipeline = pipe[0]
        self.start_optimizer = pipe[1]
        self.middle_optimizers = pipe[2]
        self.end_optimizer = pipe[3]

        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(project="R1Zero")

    def _init_pipeline(
        self,
    ) -> Tuple[Pipeline, optim.Optimizer, List[optim.Optimizer], optim.Optimizer]:
        qwen = Qwen2ForCausalLM.from_pretrained(self.hf_model_name)
        start_stage, middle_stages, end_stage = split_models_to_stages(
            qwen, len(self.devices)
        )
        start_stage, middle_stages, end_stage = stages_to_device(
            start_stage, middle_stages, end_stage, self.devices, self.loss_device
        )
        start_optimizer, middle_optimizers, end_optimizer = get_optimizers(
            start_stage,
            middle_stages,
            end_stage,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        pipeline = Pipeline(start_stage, middle_stages, end_stage, self.microbatch_size)
        return pipeline, start_optimizer, middle_optimizers, end_optimizer

    def train(
        self,
        group_size: int,
        epsilon: float,
        beta: float,
        epochs: int,
        grpo_iterations: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        reference_update_fraction: float = 1.0,
        chunk_size: int = 2,
    ):
        """
        Train the model using the GRPO algorithm.

        Args:
            group_size: The size of the group to use for GRPO.
            epsilon: The epsilon value for GRPO.
            beta: The beta value for GRPO.
            epochs: The number of epochs to train for.
            grpo_iterations: The number of GRPO iterations to perform.
            train_dataset: The dataset to use for training.
            val_dataset: The dataset to use for validation.
            reference_update_fraction: The fraction of the dataset to use until the reference model is updated.
        """
        policy_path = f"{self.project_dir}/policy"
        os.makedirs(policy_path, exist_ok=True)
        for epoch in range(epochs):
            if self.verbose:
                print(f"Epoch {epoch}")
            self.pipeline.save_full_model(f"{policy_path}/epoch_{epoch}")
            reference_model = Qwen2ForCausalLM.from_pretrained(
                f"{policy_path}/epoch_{epoch}"
            )
            reference_model.eval()
            last_step_for_wandb = 0
            for j in range(0, len(train_dataset), self.batch_size):
                batch = train_dataset[j : j + self.batch_size]
                questions = batch["question"]
                answers = batch["answer"]
                verify_types = batch["verify_type"]
                old_model = self._get_old_model()
                chats = []
                group_rewards = []
                group_skip_pad = []
                group_skip_first = []
                for i in range(len(questions)):
                    question = questions[i]
                    answer = answers[i]
                    verify_type = verify_types[i]
                    print(question)
                    (
                        outputs,
                        rewards,
                        skip_pad,
                        skip_first,
                    ) = self._get_outputs_rewards_skips(
                        old_model, question, answer, verify_type, group_size, chunk_size
                    )
                    print(f"Sum of rewards: {sum(rewards)}")
                    if self.use_wandb:
                        wandb.log(
                            {"avg_group_reward": sum(rewards) / len(rewards)},
                            step=epoch * len(train_dataset) + j + i,
                        )
                        last_step_for_wandb = epoch * len(train_dataset) + j + i
                    if sum(rewards) == 0 or sum(rewards) == len(rewards):
                        print(
                            f"Rewards are all 0 or 1 for {question} with cumulative reward: {sum(rewards)}, skipping"
                        )
                        continue
                    chats.extend(outputs)
                    group_rewards.append(rewards)
                    group_skip_pad.extend(skip_pad)
                    group_skip_first.extend(skip_first)

                if len(chats) == 0:
                    print("No chats to process, skipping batch")
                    continue

                # remove pad_tokens for proper attention masks
                chats = self._clean_chats(chats)
                tokenized = self.tokenizer(chats, padding=True, return_tensors="pt")
                input_ids = tokenized.input_ids.to(self.devices[0])
                attention_mask = tokenized.attention_mask.to(self.devices[0])
                old_model.to(self.devices[0])
                # get the logprobs of the old and reference models
                logprobs_old = self._get_logprobs(old_model, input_ids, attention_mask)
                old_model.to(self.idle_device)
                reference_model.to(self.devices[0])
                logprobs_ref = self._get_logprobs(
                    reference_model, input_ids, attention_mask
                )
                reference_model.to(self.idle_device)

                logprobs_old, advantages = self._concat_and_skip_logprobs_advantages(
                    logprobs_old, group_skip_pad, group_skip_first, group_rewards
                )
                logprobs_ref = self._concat_and_skip_logprobs(
                    logprobs_ref, group_skip_pad, group_skip_first
                )
                advantages = advantages  # .to(self.loss_device)

                # load the input_ids to the entry device of the pipeline
                # input_ids = input_ids.to(self.devices[0])
                # attention_mask = attention_mask.to(self.devices[0])
                for _ in range(grpo_iterations):
                    if self.verbose:
                        print("Start of next GRPO iteration")
                    # zero grad before chunked backward
                    self.start_optimizer.zero_grad()
                    for opt in self.middle_optimizers:
                        opt.zero_grad()
                    self.end_optimizer.zero_grad()

                    total_loss_value = 0.0
                    curr_flat_idx = 0
                    for chunk_idx in range(0, input_ids.shape[0], chunk_size):
                        logprobs_theta_chunk = self.pipeline.process_batch(
                            input_ids[chunk_idx : chunk_idx + chunk_size],
                            attention_mask[chunk_idx : chunk_idx + chunk_size],
                        )
                        logprobs_theta_chunk = self._concat_and_skip_logprobs(
                            logprobs_theta_chunk,
                            group_skip_pad[chunk_idx : chunk_idx + chunk_size],
                            group_skip_first[chunk_idx : chunk_idx + chunk_size],
                        )

                        start = curr_flat_idx
                        end = start + logprobs_theta_chunk.shape[0]

                        logprobs_old_chunk = logprobs_old[start:end].to(
                            self.loss_device
                        )
                        logprobs_ref_chunk = logprobs_ref[start:end].to(
                            self.loss_device
                        )
                        advantages_chunk = advantages[start:end].to(self.loss_device)

                        loss_chunk = grpo_loss(
                            logprobs_theta_chunk,
                            logprobs_old_chunk,
                            logprobs_ref_chunk,
                            advantages_chunk,
                            epsilon,
                            beta,
                        )

                        loss_chunk.backward(retain_graph=True)
                        total_loss_value += loss_chunk.item()
                        curr_flat_idx += logprobs_theta_chunk.shape[0]

                        del logprobs_theta_chunk
                        del logprobs_old_chunk
                        del logprobs_ref_chunk
                        del advantages_chunk

                    avg_loss_value = total_loss_value / input_ids.shape[0]
                    print(f"Average Loss for this iteration: {avg_loss_value:.4f}")
                    if self.use_wandb:
                        wandb.log({"loss": avg_loss_value}, step=last_step_for_wandb)

                    self.start_optimizer.step()
                    for opt in self.middle_optimizers:
                        opt.step()
                    self.end_optimizer.step()

                del logprobs_old
                del logprobs_ref
                del input_ids
                del attention_mask
                del tokenized

                # stop inner loop earlier eventually before the full dataset is processed
                if j > reference_update_fraction * len(train_dataset):
                    break

    def _get_outputs_rewards_skips(
        self,
        old_model: Qwen2ForCausalLM,
        question: str,
        answer: str,
        verify_type: str,
        group_size: int,
        chunk_size: int,
    ) -> Tuple[List[str], List[float], List[int], List[int]]:
        n_passes = group_size // chunk_size
        all_outputs = []
        all_rewards = []
        all_skip_pad = []
        all_skip_first = []
        for i in range(n_passes):
            tokenized = self._tokenize(question, repeat=chunk_size)
            input_ids = tokenized.input_ids.to(self.devices[0])
            attention_mask = tokenized.attention_mask.to(self.devices[0])
            with torch.no_grad():
                outputs = old_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1100,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

            if self.verbose:
                if i % 3 == 0:
                    outputs_clean = self._clean_chats(outputs)
                    print(outputs_clean[0].replace("\n\n", "\n") + "\n")

            rewards = self._compute_rewards(outputs, answer, verify_type)
            input_ids = input_ids.to("cpu")
            attention_mask = attention_mask.to("cpu")
            # skip first counts the pad tokens, this would also be possible to count the amount of 0 in the attention mask for each input
            skip_pad = [
                len(attention_mask[i]) - attention_mask[i].sum()
                for i in range(len(attention_mask))
            ]
            # skip first is for skipping the input/question tokens
            skip_first = [len(input_ids[i]) for i in range(len(input_ids))]

            all_outputs.extend(outputs)
            all_rewards.extend(rewards)
            all_skip_pad.extend(skip_pad)
            all_skip_first.extend(skip_first)
        return all_outputs, all_rewards, all_skip_pad, all_skip_first

    def _get_old_model(self):
        old_model = self.pipeline.copy_shallow_model()
        old_model.to(self.devices[0])
        return old_model

    def _optimize(self, loss: torch.Tensor):
        self.start_optimizer.zero_grad()
        for optimizer in self.middle_optimizers:
            optimizer.zero_grad()
        self.end_optimizer.zero_grad()
        print("Backward")
        loss.backward()
        print("Backward done")
        self.start_optimizer.step()
        for optimizer in self.middle_optimizers:
            optimizer.step()
        self.end_optimizer.step()
        loss.detach_()

    def _concat_and_skip_logprobs(
        self, logprobs: torch.Tensor, skip_pad: List[int], skip_first: List[int]
    ):
        # logprobs is of shape [batch_size, seq_len, vocab_size]
        # skip_first is of shape [batch_size]
        # return is of shape [batch_size * seq_len - sum(skip_first), vocab_size]
        bs, seq_len, vocab_size = logprobs.shape
        result = []
        for i in range(bs):
            sequence_logprobs = logprobs[i, skip_pad[i] + skip_first[i] :, :]
            result.append(sequence_logprobs)
        return torch.cat(result, dim=0)

    def _concat_and_skip_logprobs_advantages(
        self,
        logprobs: torch.Tensor,
        skip_pad: List[int],
        skip_first: List[int],
        rewards: List[List[float]],
    ):
        # same as above but with advantages, can be refactored in future
        bs, seq_len, vocab_size = logprobs.shape
        group_size = len(rewards[0])
        result = []
        advantages = []
        for i in range(bs):
            sequence_logprobs = logprobs[i, skip_pad[i] + skip_first[i] :, :]
            seq_len = sequence_logprobs.shape[0]
            group_rewards = rewards[i // group_size]
            reward = group_rewards[i % group_size]
            advantage = (reward - np.mean(group_rewards)) / (
                np.std(group_rewards) + 1e-8
            )
            advantage = torch.tensor([advantage] * seq_len)[:, None]
            result.append(sequence_logprobs)
            advantages.append(advantage)
        return torch.cat(result, dim=0), torch.cat(advantages, dim=0)

    def _get_logprobs(
        self,
        model: Qwen2ForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int = 1,
    ):
        all_logprobs = []
        for i in range(0, input_ids.shape[0], batch_size):
            microbatch_ids = input_ids[i : i + batch_size]
            microbatch_attention_mask = attention_mask[i : i + batch_size]
            with torch.no_grad():
                logprobs_old = model.forward(
                    microbatch_ids, attention_mask=microbatch_attention_mask
                ).logits.to("cpu")
            all_logprobs.append(logprobs_old)
        return torch.cat(all_logprobs, dim=0)

    def _tokenize(self, question: str, repeat: int = 1):
        chat_formatted = self.format_chat(question)
        chat_formatted = [chat_formatted] * repeat
        tokenized = self.tokenizer(chat_formatted, return_tensors="pt")
        return tokenized

    def format_chat(self, question: str):
        system_prompt = "<|im_start|>system\nYou are a helpful assistant. Think first about the reasoning process and then provide the answer.<|im_end|>\n"
        user_prompt = f"""<|im_start|>user\n {question}. Show your thinking in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer>4</answer>.<|im_end|>\n<|im_start|>assistant\n<think>Let me solve this step by step:\n"""
        chat_formatted = system_prompt + user_prompt
        return chat_formatted

    def _clean_chats(self, chats: List[str]):
        return [chat.replace(self.tokenizer.pad_token, "") for chat in chats]

    def _compute_rewards(self, outputs: List[str], answer: str, verify_type: str):
        return get_rewards(outputs, answer, verify_type)
