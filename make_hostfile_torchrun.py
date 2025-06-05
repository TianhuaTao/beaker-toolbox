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
    parser = argparse.ArgumentParser(
        description="Gather hostnames from all ranks and write a hostfile"
    )
    parser.add_argument(
        "output",
        help="Name of the subdirectory under /workspace/beaker_jobs to write hostfile",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Distributed backend to use (default: NCCL if GPUs are available, else GLOO)",
    )
    parser.add_argument(
        "--rank0-only",
        action="store_true",
        help="If set, only rank 0 will write the hostfile. Otherwise, all ranks will write their own hostfile.",
    )
    args = parser.parse_args()

    # Determine backend
    if args.backend:
        backend = args.backend
    else:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Initialize process group using environment variables set by torchrun
    dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    # Gather hostnames from all ranks
    hostnames = [None] * world_size
    dist.all_gather_object(hostnames, hostname)

    # Only rank 0 writes the hostfile
    if rank == 0 or not args.rank0_only:
        # Deduplicate while preserving order
        unique_hosts = []
        for h in hostnames:
            if h not in unique_hosts:
                unique_hosts.append(h)

        out_dir = os.path.join("/workspace/beaker_jobs", args.output)
        os.makedirs(out_dir, exist_ok=True)
        hostfile_path = os.path.join(out_dir, "hostfile")

        with open(hostfile_path, "w") as f:
            for h in unique_hosts:
                f.write(f"{h}\n")

        print(f"Hostfile written to {hostfile_path}")
        
        
        # create timestamp file
        timestamp_path = os.path.join(out_dir, "latest-timestamp.txt")
        with open(timestamp_path, "w") as f:
            now = datetime.now(ZoneInfo("America/Los_Angeles"))
            f.write(now.strftime("%Y-%m-%d-%H-%M-%S")+'\n')
            

    # Wait for all ranks to finish
    dist.barrier()

if __name__ == "__main__":
    main()
