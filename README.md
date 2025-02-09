# Pipeline Parallel Reinforcement Learning Training (DeepSeek R1 style)

> **Warning**: This project is currently in a very early experimental stage and not yet ready for use. All stuff is subject to significant changes and improvements. This implementation is primarily a learning project to rebuild the DeepSeek R1 training pipeline from scratch.

## Current State

This project implements a very basic version of pipeline-parallel training for language models using Group-Relative Policy Optimization (GRPO) in the style of R1 Zero (https://arxiv.org/pdf/2501.12948). The current implementation:

- Has only been tested with Qwen2.5 0.5B & 1.5B Instruct model on 4xL4 and 2xL4 GPUs
- Has not yet produced "good" results for simple tasks (e.g. multiplication, character counting, task scheduling) but is able to train at get somehwat better (not yet tested a lot, I think it's mainly a matter of hyperparameters. As soon as I have a good training run, I'll update this section)
- Is not optimized for performance (e.g. the parallelism is implemented using threads, also the backward pass is not parallelized. All this is very inefficient and is subject to change)
- Lacks many important features (e.g. parallel generation/sampling, validation pipeline, excessive logging, tensor parallelism support)

### Key Features

- GRPO implementation following https://arxiv.org/pdf/2402.03300 (except that here "hard" validations are used, so rule based validations/rewards like in DeepSeek R1)
- Support for Qwen2 models
- Simple numerical reward functions


### Performance Issues/Feature Gaps
- Generation/Sampling is not parallelized (all sampling happens on a single GPU)
- Reference and old policy models are processed on single GPUs rather than being pipeline parallel/tensor parallel
- Limited optimization for memory efficiency
- No tensor parallelism support yet
- No multi-node distributed training support yet
- No good checkpointing/failure recovery
- No validation pipeline
- Limited model architecture support (only Qwen2.x is supported)
- Limited reward function types (only simple numerical comparisons)


## Requirements

- PyTorch
- Transformers (v4.47.1)
- Datasets
- CUDA-capable GPUs (minimum 2)

## Installation

```bash
pip install torch transformers=4.47.1 accelerate datasets
```

## Basic Usage

```python
from pipeline.trainer import GRPOTrainer

trainer = GRPOTrainer(
    hf_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    learning_rate=1e-6,
    weight_decay=0.001,
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    batch_size=16,
    microbatch_size=1,
    project_dir="test",
    loss_device="cuda:2",
    verbose=True
)

trainer.train(
    group_size=10,
    epsilon=0.125,
    beta=0.001,
    epochs=2,
    grpo_iterations=10,
    train_dataset=dataset, 
    val_dataset=dataset, # validation dataset is not yet used
    reference_update_fraction=1.0,
    chunk_size=1)
```

Example dataset:
```python
from datasets import Dataset

data = [
{'question': 'Below is a list of tasks, each with a duration (in days) and a set of prerequisite tasks. Tasks can only start once their prerequisites finish, but they can run in parallel if no dependencies are blocking them. What is the smallest total number of days required to complete every task?\nTasks:\nT0(duration=4, deps=[None])\nT1(duration=1, deps=[None])\nT2(duration=2, deps=[T0])\nT3(duration=10, deps=[T0])\n', 'answer': '14', 'verify_type': 'LAST_NUMBER'},
{'question': 'Below is a list of tasks, each with a duration (in days) and a set of prerequisite tasks. Tasks can only start once their prerequisites finish, but they can run in parallel if no dependencies are blocking them. What is the smallest total number of days required to complete every task?\nTasks:\nT0(duration=1, deps=[None])\nT1(duration=3, deps=[T0])\nT2(duration=2, deps=[T1, T0])\nT3(duration=5, deps=[None])\nT4(duration=10, deps=[T0, T1, T3, T2])\n', 'answer': '16', 'verify_type': 'LAST_NUMBER'},
]
dataset = Dataset.from_list(data)
```

## Implementation Details

### Pipeline Structure
The model is split across GPUs into:
- Start Stage: Embeddings + initial transformer blocks
- Middle Stage(s): Additional transformer blocks (if >2 GPUs)
- End Stage: Final transformer blocks + LM head

### Current Pipeline Limitations
1. Only the target policy model uses pipeline parallelism
2. Reference and old policy models run on single GPUs
3. Sampling/generation is not parallelized
4. Limited overall memory optimization

### Why Build from Scratch?
This project intentionally avoids using established frameworks like DeepSpeed to:
1. Better understand pipeline parallelism implementation details
2. Learn GRPO implementation nuances
3. Have full control over optimization strategies
4. Serve as a learning resource for others

However, the goal is to eventually achieve production-level efficiency while maintaining this educational value.

## Contributing

This project is open to contributions, but please note its experimental nature. Feel free to open issues or PRs for:
- Bug fixes
- Performance improvements
- New features from the roadmap
- Documentation improvements

## License

Apache 2.0