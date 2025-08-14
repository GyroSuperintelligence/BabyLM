# kernel/gpt_oss/torch/utils.py
import os
import torch
import torch.distributed as dist


def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed() -> torch.device:
    """
    CPU-only initializer compatible with gpt_oss.torch.utils.init_distributed.
    - never initializes NCCL
    - never calls torch.cuda.*
    - world_size=1, rank=0
    - returns CPU device
    """
    # force single-process CPU path
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    if dist.is_initialized():
        dist.destroy_process_group()  # be safe if caller did something earlier
    device = torch.device("cpu")
    suppress_output(rank=0)
    return device
