#!/usr/bin/env python3
"""
Script to gather all ranks' hostnames in a torchrun job and write them
to /workspace/beaker_jobs/${OUTPUT}/hostfile
"""
import os
import socket
import argparse
import torch
import torch.distributed as dist
from datetime import datetime
from zoneinfo import ZoneInfo

def main():

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Initialize process group using environment variables set by torchrun
    dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()


    # Wait for all ranks to finish
    dist.barrier()

    print(f"Rank {rank}/{world_size} complete")

if __name__ == "__main__":
    main()
