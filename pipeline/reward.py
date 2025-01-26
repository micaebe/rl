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

import re
from typing import List


def get_rewards(outputs: List[str], answer: str, verify_type: str):
    rewards = []
    for i, output in enumerate(outputs):
        if verify_type == "BOXED_NUMERICAL":
            rewards.append(calc_numeric_reward(output, answer))
        elif verify_type == "LAST_NUMBER":
            rewards.append(calc_last_number_reward(output, answer))
        else:
            raise ValueError(f"Unknown verify type: {verify_type}")
    return rewards


def calc_numeric_reward(output: str, answer: str):
    try:
        extracted_number_output = extract_boxed(output)
        return 1 if float(extracted_number_output) == float(answer) else 0
    except:
        return 0


def calc_last_number_reward(output: str, answer: str):
    try:
        extracted_number_output = extract_last_number(output)
        return 1 if float(extracted_number_output) == float(answer) else 0
    except:
        return 0


def extract_last_number(s: str) -> str | None:
    pattern = r"[+-]?\d+(?:\.\d+)?"

    matches = re.findall(pattern, s)
    if matches:
        return matches[-1]
    return None


def extract_boxed(text):
    start_marker = r"\boxed{"
    start = text.rfind(start_marker)
    if start == -1:
        return ""

    start_idx = start + len(start_marker) - 1
    balance = 1
    end_idx = start_idx

    while end_idx < len(text) - 1 and balance > 0:
        end_idx += 1
        char = text[end_idx]
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1

    if balance == 0:
        content = text[start_idx + 1 : end_idx]
    else:
        content = text[start_idx + 1 : end_idx + 1]

    if len(content) >= 2 and content.startswith("{") and content.endswith("}"):
        return content[1:-1]
    else:
        return content
