# OLMo Training with Layer-Wise Scaling, Group Query Attention, and FSDP

This repository contains an enhanced implementation for training the OLMo language model with three key improvements:

1.  **Layer-Wise Scaling (LWS):** Applies different scaling factors to different layers based on their depth, improving gradient flow and training dynamics.
2.  **Group Query Attention (GQA):** Reduces memory requirements and computational costs by sharing key and value projections across multiple query heads.
3.  **Fully Sharded Data Parallel (FSDP):** Enables efficient distributed training across multiple GPUs and nodes by sharding model parameters, gradients, and optimizer states.

The implementation builds upon the OLMo core library and provides a robust, scalable training pipeline.

## Project Structure

```

olmo-training/
├── olmo\_core/              \# Core OLMo implementation (already exists)
├── config/
│   ├── base\_config.yaml    \# Base configuration
│   └── fsdp\_config.yaml    \# FSDP-specific configurations
├── scripts/
│   ├── prepare\_data.py     \# Data preprocessing script
│   └── launch\_fsdp.py      \# Multi-node launch script
├── src/
│   ├── data/               \# Data loading utilities
│   ├── model/              \# Model building with LWS and GQA
│   ├── training/           \# Training utilities with FSDP support
│   └── utils/              \# General utilities
├── train.py                \# Main training entry point
└── README.md               \# This file

````

## Setup and Installation

**Prerequisites**

* Python 3.8+
* PyTorch 2.0+
* CUDA-compatible GPU(s) for efficient training

**Installation**

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/yourusername/olmo-training.git](https://github.com/yourusername/olmo-training.git)
    cd olmo-training
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up OLMo core (if not already installed):**

    `# Option 1: If OLMo core is installed via pip`
    ```bash
    pip install olmo-core
    ```

    `# Option 2: If OLMo core is cloned from GitHub`
    ```bash
    git clone [https://github.com/allenai/OLMo-core.git](https://github.com/allenai/OLMo-core.git)
    ln -s OLMo-core/src/olmo_core olmo_core
    ```

## Data Preparation

Use the provided script to download and preprocess Wikipedia data:

```bash
python scripts/prepare_data.py --output-dir ./data --sequence-length 1024
````

Advanced options:

```bash
python scripts/prepare_data.py --config config/base_config.yaml --max-articles 10000
```

## Training

**Single-GPU Training**

For simple experiments or testing, run training on a single GPU:

```bash
python train.py --config config/base_config.yaml --inference-prompt "The universe is"
```

**Multi-GPU Training with FSDP**

For distributed training using FSDP:

```bash
python train.py --config config/base_config.yaml --fsdp-config config/fsdp_config.yaml
```

Using the provided launch script:

```bash
python scripts/launch_fsdp.py --config config/base_config.yaml --nodes 2 --gpus-per-node 4
```

**SLURM-Based Training**

For cluster environments with SLURM:

```bash
python scripts/launch_fsdp.py --config config/base_config.yaml --launcher slurm --nodes 4
```

## Key Features

**Layer-Wise Scaling (LWS)**

LWS applies different scaling factors to different layers of the Transformer, which helps with gradient flow during training. Configure LWS in your configuration file:

```yaml
# Layer-Wise Scaling parameters
fnn_scalars: [0.5, 4.0]  # [min_scale, max_scale] for feed-forward networks
qkv_scalars: [0.5, 1.0]  # [min_scale, max_scale] for query-key-value projections
output_scalars: [0.8, 1.2] # [min_scale, max_scale] for output projections
```

**Group Query Attention (GQA)**

GQA reduces memory usage and computation by sharing key-value heads across multiple query heads. Configure GQA in your configuration file:

```yaml
# Attention configuration
n_heads: 12       # Number of query heads
n_kv_heads: 3     # Number of key-value heads (should divide n_heads)
```

**Fully Sharded Data Parallel (FSDP)**

FSDP enables efficient distributed training by sharding model parameters, gradients, and optimizer states across multiple GPUs. Configure FSDP in `fsdp_config.yaml`:

```yaml
# FSDP configuration
mixed_precision: true     # Whether to use mixed precision
shard_strategy: "FULL_SHARD" # Sharding strategy
cpu_offload: false         # Whether to offload parameters to CPU
```

## Monitoring and Evaluation

**Weights & Biases Integration**

Enable Weights & Biases logging by setting `use_wandb: true` in your configuration or using the `--wandb` flag:

```bash
python train.py --config config/base_config.yaml --wandb
```

**Text Generation During Training**

Monitor training progress with periodic text generation using custom prompts:

```yaml
# In your config file
inference_interval: 50
inference_prompt: "The universe is"
max_new_tokens: 50
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

  * The **Allen Institute for AI** for creating the OLMo language model and ecosystem
  * The **PyTorch team** for the FSDP implementation

<!-- end list -->

```
```