#!/usr/bin/env python3
"""
Wrapper script to use torch.distributed.launch instead of torchrun for older PyTorch versions.
Converts torchrun-style arguments to torch.distributed.launch format.
"""

import subprocess
import sys
import argparse


def main():
    """Convert torchrun arguments to torch.distributed.launch format."""
    
    parser = argparse.ArgumentParser(description="torchrun wrapper for older PyTorch")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Processes per node")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port")
    parser.add_argument("training_script", help="Training script to run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments for script")
    
    args = parser.parse_args()
    
    # Build torch.distributed.launch command
    cmd = [
        sys.executable,
        "-m", "torch.distributed.launch",
        "--nproc_per_node", str(args.nproc_per_node),
        "--nnodes", str(args.nnodes),
        "--node_rank", str(args.node_rank),
        "--master_addr", args.master_addr,
        "--master_port", str(args.master_port),
        args.training_script,
    ] + args.script_args
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
