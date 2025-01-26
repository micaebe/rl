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

import torch


def grpo_loss(
    logprobs_theta: torch.Tensor,
    logprobs_old: torch.Tensor,
    logprobs_ref: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.1,
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Computes the GRPO loss.

    Args:
      logprobs_theta: shape [batch, seq_len, vocab_size], current policy log-probs.
      logprobs_old:   shape [batch, seq_len, vocab_size], old policy log-probs.
      logprobs_ref:   shape [batch, seq_len, vocab_size], reference policy log-probs.
      advantages:     shape [batch, seq_len, 1], group-relative advantages.
      epsilon:        clipping parameter.
      beta:           coefficient for the KL term.
    Returns:
      The GRPO loss.
    """

    # PPO-like clipped objective, averaged over batch and tokens
    ratio = torch.exp(logprobs_theta - logprobs_old)  # curr_policy / old_policy
    first_term = ratio * advantages
    second_term = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    ppo_term = torch.mean(torch.min(first_term, second_term))

    # unbiased KL estimator
    ratio_ref = torch.exp(logprobs_ref - logprobs_theta)
    kl_est = ratio_ref - 1.0 - torch.log(ratio_ref)
    kl_term = torch.mean(kl_est)

    # GRPO objective: maximize J = PPO_term - beta * KL
    # therefore minimize -J = -PPO_term + beta * KL via gradient descent
    loss = -ppo_term + beta * kl_term
    return loss
