# OLMo 190M Training ON SNELLIUS

This repository contains a clean, modular implementation for training a lightweight 190M parameter OLMo model on Wikipedia data. The codebase is structured to provide a complete training pipeline with command-line configuration options.

## Project Structure

```
clean_train/
├── configs/            
│   ├── argparser.py     # Command-line argument parsing
│   ├── model_configs.py 
│   └── optimizer_configs.py 
├── data/              
│   ├── dataset.py      # Dataset preparation
│   ├── dataloader.py   
│   ├── load_dataset.py 
│   ├── tokenizer.py    
│   └── validate_data_prep.py 
├── training/         
│   ├── trainer.py      # Training execution
│   └── callbacks/     
│       └── inference_callback.py 
├── utils/             
│   ├── environment.py  # Environment setup (e.g., API keys, cache directories)
│   ├── paths.py        # Path management for data and model outputs
│   └── token_0_handling.py 
├── data_prep.py        # Run to download and tokenize data 
└── train.py            # Run to start training
```

## Setting it up

### `configs/argparser.py`
This file handles command-line argument parsing for both training (`train.py`) and data preparation (`data_prep.py`). It defines the available arguments, their default values, and help descriptions. This allows for easy configuration of training parameters, data paths, and other settings from the command line.

### `utils/paths.py`
The `paths.py` module is responsible for managing directory structures for the project. It contains functions to:
- Set up data directories for training, including a main data directory and a subdirectory for Hugging Face cache.
- Create timestamped output folders for saving model checkpoints during training.
- Set up data directories for data preparation, including the main data directory, Hugging Face cache, and a specific directory for Wikipedia data.
It ensures that all necessary directories are created if they don't exist and provides a centralized way to define and access important paths.

### `utils/environment.py`
This script sets up the necessary environment variables for the training process. Key functionalities include:
- Disabling tokenizer parallelism to avoid potential issues.
- Setting up and creating local cache directories for Hugging Face `datasets`, `transformers`, and general `huggingface` assets. This helps in managing downloaded models and datasets efficiently.
- Configuring the Weights & Biases API key for experiment tracking.

## Setting up



## Training Pipeline

The pipeline consists of two main scripts:

1. `data_prep.py`: Downloads Wikipedia data and tokenizes it using the OLMo tokenizer
2. `train.py`: Trains the OLMo 190M model with configurable parameters

### Key Features

- **Modular Architecture**: Clean separation of concerns across modules
- **Command-line Configuration**: Easy parameter adjustment via command-line arguments
- **Efficient Training**: Uses Flash Attention for faster training
- **Data Storage Management**: Automatically manages cache directories and data storage
- **WandB Integration**: Tracks metrics and generated text in Weights & Biases
- **Progress Monitoring**: Includes both console logging and real-time metrics
- **Inference Testing**: Periodically generates text from a prompt to evaluate model quality

## Setting up the Environment

To set up your environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/baroian/OLMo-custom
   cd OLMo-custom
   ```

2. **Load necessary modules (example for Snellius):**
   ```bash
   module load 2024
   module load Miniconda3/24.7.1-0
   source $EBROOTMINICONDA3/etc/profile.d/conda.sh
   # Load PyTorch separately for GPU support
   module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
   ```

3. **Create and activate the conda environment:**
   ```bash
   conda create -n olmo python=3.9
   conda activate olmo
   ```

4. **Install base dependencies:**
   ```bash
   pip install torch wandb datasets tqdm numpy transformers
   ```

5. **Install OLMo Core:**
   ```bash
   git clone https://github.com/allenai/OLMo-core.git
   cd OLMo-core
   pip install -e .[all]
   cd .. # Return to OLMo-custom directory
   ```

## Data Preparation

First, prepare the Wikipedia dataset using the data preparation script:

```bash
python clean_train/data_prep.py --sequence-length=1024 --target-tokens=300000 
```

Available command-line arguments:

```
--sequence-length INT     Sequence length for tokenization (default: 1024)
--output-file STR         Output filename for tokenized data (default: wiki_tokens.npy)
--target-tokens INT       Target number of tokens to collect (default: 300,000)
--data-dir STR            Data directory to use (default: auto-detected)
```

We don't know how many tokens per wikipedia article so we bound the amount of data donwloaded and tokenized by both, target tokens and percentage of articles (0.01 means 1% of wikipedia)

## Running the Training Job

After data preparation, run the training script:

```bash
python clean_train/train.py --batch-size=24 --micro-batch-size=4 --steps=100
```

### Command-line Arguments

```
--gpu INT                GPU ID to use (default: 0)
--steps INT              Total training steps (default: 100)
--batch-size INT         Batch size (default: 2)
--macro-batch-size INT   Micro Batch Size (default: 1)
--inference-interval INT Run inference every N steps (default: 200)
--inference-prompt STR   Prompt for inference (default: "Dutch is ")
--wandb-project STR      WandB project name (default: "olmo-training")
--wandb-name STR         WandB run name (default: "olmo-train")
```

### Understanding Batch Sizes and Gradient Accumulation

- **Batch Size**: The total number of tokens processed in a single update step (should be around 0.5M tokens)
  - Example: `--batch-size=600` means 600 × 1024 = 614,400 tokens per update step

- **Micro Batch Size**: The number of sequences that fit on a single GPU for one forward/backward pass
  - Example: `--micro-batch-size=24` means 24 × 1024 = 24,576 tokens processed in one forward/backward pass

The training system automatically handles gradient accumulation to reach the desired effective batch size across multiple micro batches.

### Example for Snellius with Slurm

1. **Allocate GPU resources:**
   ```bash
   salloc --partition=gpu_h100 --gres=gpu:h100:1 --time=00:30:00
   ```

2. **SSH into the compute node:**
   ```bash
   ssh <node_name>
   ```

3. **Load modules and run the training script:**
   ```bash
   # Load modules
   module load 2024
   module load Miniconda3/24.7.1-0
   source $EBROOTMINICONDA3/etc/profile.d/conda.sh
   module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

   # Activate conda environment
   conda activate olmo

   # Run training
   python clean_train/train.py --batch-size=24 --steps=1000
   ```

## GPU Performance

- **GTX 3090 (24GB)**: ~20k tokens per second (TPS)
- **A100 (40GB)**: ~75k TPS (1h = 128 SBUs)
- **H100 (95GB)**: ~135k TPS (1h = 192 SBUs)

H100 is approximately 1.5 times more expensive but 1.8 times more efficient than A100, making it worth the investment for larger training runs.

## Outputs

- Model checkpoints saved in the configured data directory
- Training logs stored in the same directory
- Metrics tracked in Weights & Biases

## Requirements

- Python 3.9+
- PyTorch 2.1.2+ (matching the loaded module/CUDA version)
- OLMo Core library (`ai2-olmo`, installed from source)
- Flash Attention (included via `OLMo-core[all]`)
- Hugging Face libraries (`datasets`, `transformers`)
- WandB for experiment tracking
- Utility libraries (`tqdm`, `numpy`, etc.)

## Acknowledgements

This project builds upon the [OLMo Core](https://github.com/allenai/OLMo-core) library developed by the Allen Institute for Artificial Intelligence.
