#!/bin/bash

#SBATCH --job-name=olmo_ddp
#SBATCH --output=slurm_outputs/olmo_ddp_%j.out
#SBATCH --partition=gpu_h100
#SBATCH --nodes=2                    # 2 nodes (change as needed)
#SBATCH --gres=gpu:h100:4            # 4 GPUs *per node*
#SBATCH --ntasks-per-node=1          # one launcher per node (torchrun will fork per-GPU)
#SBATCH --cpus-per-task=8            # threads dataloader etc.
#SBATCH --time=8:00:00

# Navigate to the script's own directory.
# This ensures that paths to venv and train.py are resolved correctly,
# assuming they are in the same directory as this script.
echo "Initial working directory: $(pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "Changing to script directory: $SCRIPT_DIR"
#cd "$SCRIPT_DIR" || { echo "Failed to change to script's directory: $SCRIPT_DIR"; exit 1; }
echo "Current working directory: $(pwd)"

module load 2024
module load CUDA/12.6.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Optional NCCL tuning (RoCE / IB); tweak for Snellius network hardware
export NCCL_DEBUG=warn
## Remove interface restrictions; let NCCL auto-select.
#export NCCL_IB_HCA=mlx5_0,mlx5_1
#export NCCL_SOCKET_IFNAME=ib0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Tell OLMo that the nodes share a filesystem so checkpointing works across ranks.
export OLMO_SHARED_FS=1

# ─── 1. Torch-distributed variables ───────────────────────────────────────
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
NODE_RANK=$SLURM_PROCID   # Rank of this launcher within the multi-node job
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500   # pick any free port, avoid 2-3xxxx range SLURM uses

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"

# Activate the virtual environment
echo "Activating virtual environment..."
source new_venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Run the training script
echo "Running Python training script..."
#python train.py --config=config

# ─── 3. Launch: one torchrun *per node* via srun ───────────────────────────
#   Each srun task (there is one per node) calls torchrun, which then forks
#   one subprocess per GPU on its local node.  This ensures every node joins
#   the rendez-vous and avoids startup hangs.

# Debug: show network interfaces inside each launcher
echo "Available network interfaces:" && ip -o link || true

srun --label --output=slurm_outputs/olmo_ddp_%j_%t.out \
  --gpus-per-task=$GPUS_PER_NODE \
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --distributed

echo "Training script finished." 