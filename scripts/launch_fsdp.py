#!/usr/bin/env python3
"""
Multi-node launch script for FSDP training of OLMo.
"""

import os
import sys
import argparse
import yaml
import logging
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch FSDP training for OLMo")
    
    # Distributed training configuration
    parser.add_argument(
        "--nodes", 
        type=int, 
        default=1, 
        help="Number of nodes"
    )
    parser.add_argument(
        "--gpus-per-node", 
        type=int, 
        default=8, 
        help="Number of GPUs per node"
    )
    parser.add_argument(
        "--node-rank", 
        type=int, 
        default=0, 
        help="Rank of this node"
    )
    parser.add_argument(
        "--master-address", 
        type=str, 
        default="localhost", 
        help="Address of master node"
    )
    parser.add_argument(
        "--master-port", 
        type=str, 
        default="29500", 
        help="Port for master node"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["nccl", "gloo"], 
        default="nccl", 
        help="Distributed backend"
    )

    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1, 
        help="Local rank passed by torch.distributed.launch"
    )
    
    # Model and training configuration
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--fsdp-config", 
        type=str, 
        default=None, 
        help="Path to FSDP configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output", 
        help="Directory for outputs"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to checkpoint to resume from"
    )
    
    # Launch configuration
    parser.add_argument(
        "--no-python", 
        action="store_true", 
        help="Do not prepend 'python' to command"
    )
    parser.add_argument(
        "--script", 
        type=str, 
        default="train.py", 
        help="Script to run"
    )
    parser.add_argument(
        "--launcher", 
        type=str, 
        choices=["torch", "slurm"], 
        default="torch", 
        help="Launcher to use"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_slurm_nodelist() -> List[str]:
    """Get list of nodes from SLURM environment."""
    if "SLURM_JOB_NODELIST" not in os.environ:
        raise ValueError("SLURM_JOB_NODELIST not found in environment")
    
    # This is a simplified implementation - a real one would use scontrol to expand the nodelist
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    
    # Execute scontrol to expand the node list
    cmd = f"scontrol show hostnames {nodelist}"
    result = subprocess.check_output(cmd, shell=True).decode().splitlines()
    
    return result

def get_world_size(nodes: int, gpus_per_node: int) -> int:
    """Get world size for distributed training."""
    return nodes * gpus_per_node

def build_torch_distributed_launch_command(args):
    """Build command for torch.distributed.launch."""
    world_size = get_world_size(args.nodes, args.gpus_per_node)
    
    cmd = []
    
    if not args.no_python:
        cmd.append(sys.executable)
    
    # With:
    cmd.extend([
        "-m", "torch.distributed.run",
        f"--nproc_per_node={args.gpus_per_node}",
        f"--nnodes={args.nodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_address}",
        f"--master_port={args.master_port}"
    ])
    
    # Add script and its arguments
    cmd.append(args.script)
    cmd.extend([
        f"--config={args.config}",
        f"--use-fsdp",
        f"--output-dir={args.output_dir}",
        f"--distributed-backend={args.backend}",
        f"--world-size={world_size}"
    ])
    
    # Add FSDP config if specified
    if args.fsdp_config:
        cmd.append(f"--fsdp-config={args.fsdp_config}")
    
    # Add checkpoint if specified
    if args.checkpoint:
        cmd.append(f"--checkpoint={args.checkpoint}")
    
    return cmd

def build_slurm_launch_command(args):
    """Build command for SLURM-aware launch."""
    world_size = get_world_size(args.nodes, args.gpus_per_node)
    
    # Get node list from SLURM
    try:
        nodes = get_slurm_nodelist()
    except Exception as e:
        print(f"Error getting SLURM nodelist: {e}")
        return None
    
    if len(nodes) != args.nodes:
        print(f"Warning: SLURM allocated {len(nodes)} nodes, but {args.nodes} were requested")
    
    # Set master address to first node
    master_addr = nodes[0]
    
    cmd = []
    
    if not args.no_python:
        cmd.append(sys.executable)
    
    cmd.extend([
        "-m", "olmo_core.launch.slurm",  # Use OLMo's SLURM launcher if available
        f"--nnodes={args.nodes}",
        f"--ntasks_per_node={args.gpus_per_node}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={args.master_port}"
    ])
    
    # Add script and its arguments
    cmd.append(args.script)
    cmd.extend([
        f"--config={args.config}",
        f"--use-fsdp",
        f"--output-dir={args.output_dir}",
        f"--distributed-backend={args.backend}",
        f"--world-size={world_size}"
    ])
    
    # Add FSDP config if specified
    if args.fsdp_config:
        cmd.append(f"--fsdp-config={args.fsdp_config}")
    
    # Add checkpoint if specified
    if args.checkpoint:
        cmd.append(f"--checkpoint={args.checkpoint}")
    
    return cmd

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration if provided
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Choose launcher
    if args.launcher == "torch":
        cmd = build_torch_distributed_launch_command(args)
    elif args.launcher == "slurm":
        cmd = build_slurm_launch_command(args)
        if cmd is None:
            print("Failed to build SLURM launch command")
            return 1
    else:
        print(f"Unsupported launcher: {args.launcher}")
        return 1
    
    # Print command
    print("Launching with command:")
    print(" ".join(cmd))
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())