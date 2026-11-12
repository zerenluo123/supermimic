"""Data-parallel training across GPUs, launched with torchrun (one process per GPU).

Each rank owns its own simulator, replay buffer and reset sampler; the world model,
actor and critic are replicated: rank 0's initial weights are broadcast, every
optimizer step averages the gradients across ranks first (inside the captured
update), and the empirical normalizers pool their batch statistics (rsl_rl's
``EmpiricalNormalization`` all-reduces them once a process group is up). Ranks
therefore stay identical without a DDP wrapper.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def pin_local_gpu() -> None:
  """Under torchrun, show each rank only its own GPU (as ``cuda:0``), before CUDA
  initializes: the simulator and the compiled kernels then need no device index."""
  if "LOCAL_RANK" not in os.environ or os.environ.get("SM_GPU_PINNED") == "1":
    return
  local_rank = int(os.environ["LOCAL_RANK"])
  visible = os.environ.get("CUDA_VISIBLE_DEVICES")
  gpus = visible.split(",") if visible else [str(i) for i in range(64)]
  os.environ["CUDA_VISIBLE_DEVICES"] = gpus[local_rank]
  os.environ["SM_GPU_PINNED"] = "1"


def init_distributed() -> tuple[int, int]:
  """``(rank, world_size)``; starts NCCL when launched by torchrun with >1 rank."""
  world = int(os.environ.get("WORLD_SIZE", "1"))
  if world == 1:
    return 0, 1
  torch.cuda.set_device(0)
  dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
  return dist.get_rank(), world


def world_size() -> int:
  return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
  return dist.get_rank() if dist.is_initialized() else 0


def is_main() -> bool:
  return rank() == 0


@torch.no_grad()
def average_gradients(params: list[torch.Tensor]) -> None:
  """Average the gradients of ``params`` across ranks with one all-reduce.

  Capture-safe: no host sync, so it runs inside a recorded update.
  """
  if world_size() == 1:
    return
  grads = [p.grad for p in params if p.grad is not None]
  if not grads:
    return
  flat = torch.cat([g.reshape(-1) for g in grads])
  dist.all_reduce(flat, op=dist.ReduceOp.AVG)
  chunks = flat.split([g.numel() for g in grads])
  torch._foreach_copy_(
    grads, [c.view_as(g) for c, g in zip(chunks, grads, strict=True)]
  )


@torch.no_grad()
def broadcast_module(module: torch.nn.Module) -> None:
  """Copy rank 0's parameters and buffers to every rank."""
  if world_size() == 1:
    return
  for t in [*module.parameters(), *module.buffers()]:
    dist.broadcast(t.data, src=0)


def finalize() -> None:
  """Let every rank finish its last collective, then exit with NCCL still up:
  ``destroy_process_group`` hangs once the captured updates have recorded
  all-reduces, while process exit releases the communicator cleanly."""
  if dist.is_initialized():
    torch.cuda.synchronize()
    dist.barrier()
