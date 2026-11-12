"""Script to train World Model using SuperMimicRunner."""

from __future__ import annotations

import os
from dataclasses import dataclass

from supermimic.utils.distributed import pin_local_gpu

pin_local_gpu()  # under torchrun, before anything initializes CUDA

import mjlab.tasks  # noqa: E402, F401
import tyro  # noqa: E402
from mjlab.envs import ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.tasks.registry import list_tasks, load_env_cfg  # noqa: E402

import supermimic.tasks  # noqa: E402, F401
from supermimic.runners.supermimic_runner import (  # noqa: E402
  SuperMimicRunner,
  SuperMimicRunnerCfg,
)
from supermimic.scripts.train_utils import (  # noqa: E402
  create_env,
  download_registry_motions,
  setup_device,
)
from supermimic.utils.distributed import finalize, init_distributed  # noqa: E402


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  # Experiment & Environment
  num_envs: int = 4096
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  motion_file: str | None = None
  scale_lr_with_world: bool = True
  """Under torchrun with N ranks, multiply the learning rates by sqrt(N) (the
  averaged gradient of N times the data is less noisy)."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=SuperMimicRunnerCfg(),
    )


def _passed(args: list[str], flag: str) -> bool:
  """``flag`` (``--a.b-c``) is on the command line, in either spelling."""
  spellings = {flag, flag.replace("-", "_").replace("__", "--", 1)}
  return any(a.split("=", 1)[0] in spellings for a in args)


def apply_g1d_defaults(runner: SuperMimicRunnerCfg, args: list[str]) -> None:
  """The G1-D trains the deployable actor: proprio with a 10-frame history, never
  the true state. Fills in what the command line leaves at the G1 defaults and
  refuses ``--runner.actor-obs rep_state``."""
  if runner.representation == "gencoord":
    runner.representation = "g1d"
    print("[INFO] Auto-detected g1d representation from task_id")
  if runner.actor_obs == "rep_state":
    if _passed(args, "--runner.actor-obs"):
      raise ValueError(
        "--runner.actor-obs rep_state feeds the G1-D actor the true, noise-free "
        "state, which the robot does not have; train it on proprio"
      )
    runner.actor_obs = "proprio"
    print("[INFO] G1-D actor obs: proprio")
  if runner.history_length == 1 and not _passed(args, "--runner.history-length"):
    runner.history_length = 10
    print("[INFO] G1-D actor history: 10 frames")


def main() -> None:
  # Parse first argument to choose the task.
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=mjlab.TYRO_FLAGS,
  )

  device = setup_device(args.device)
  rank, world = init_distributed()
  if world > 1:
    if args.env.seed is not None:
      args.env.seed += rank
    if args.scale_lr_with_world:
      args.runner.wm.lr *= world**0.5
      args.runner.policy.lr *= world**0.5
    print(
      f"[INFO] Rank {rank}/{world}: wm lr {args.runner.wm.lr:g}, "
      f"policy lr {args.runner.policy.lr:g}, env seed {args.env.seed}"
    )
  print(f"[INFO] Loading environment: {chosen_task}")

  if args.motion_file:
    # Local motions handed straight to MotionLoader (globs/concatenates).
    if "," in args.motion_file:
      args.env.commands["motion"].motion_file = [
        os.path.expanduser(p.strip()) for p in args.motion_file.split(",")
      ]
    else:
      args.env.commands["motion"].motion_file = os.path.expanduser(args.motion_file)
    args.registry_name = None  # avoid linking a stale registry artifact for lineage
    print(f"[INFO] Using local motion file: {args.motion_file}")
  else:
    motion_file = download_registry_motions(args.registry_name)
    if motion_file is not None:
      args.env.commands["motion"].motion_file = motion_file

  # Auto-detect representation from task_id if not explicitly set
  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")
  if chosen_task.endswith("-G1D"):
    apply_g1d_defaults(args.runner, remaining_args)

  args.env.commands["motion"].sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  runner = SuperMimicRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()
  finalize()


if __name__ == "__main__":
  main()
