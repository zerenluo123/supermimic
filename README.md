# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## G1-D offline replay

Record a kept checkpoint once, in parallel on a GPU (no rendering):

```bash
uv run src/supermimic/scripts/record_g1d.py \
  --checkpoint-dir <run>/checkpoints/00060000 \
  --motion-file 'data/lafan1_g1d_v3/*.npz' --eval-tag lafan \
  --output-dir runs/replay_00060000/lafan --num-envs 128
```

For BONES use `--motion-file data/bones_v4_eval2000 --eval-tag bones2000`
and a separate output directory. Repeating the command resumes completed clips;
different weights, motion file metadata, or recording code require a new directory.
The cache contains an `index.json` and one compressed `.npz` per clip. It stores
float32 joint poses for the simulator, reference and closed-loop world model,
plus step errors and rewards, not meshes or video. Each frame uses 281 bytes
before compression (21 joints × 3 poses × 4 bytes plus metrics). Recording runs
without observation/action/reset noise and continues after failures.

Play from disk without running a policy or GPU simulation:

```bash
uv run src/supermimic/scripts/play_g1d_replay.py \
  --replay-dir runs/replay_00060000/lafan \
  --visual-mesh-dir data/g1d_visual_meshes --port 8080
```

The existing `play.py SuperMimic-GenCoord-G1D` also accepts `--replay-dir` and
`--port`. Omit `--visual-mesh-dir` to use the shipped collision meshes. Original
visual meshes and both ghosts are drawn on every playback frame; mesh complexity
and network/browser speed still affect rendering. Only the selected clip is
decompressed in memory. No client connected means playback stays idle.

The panel supports filtering, failed clips, previous/next clip, automatic next,
pause, forward/backward step, speed, a frame slider, jump to first failure,
camera follow and ghost visibility. Scrubbing pauses playback. Frame 0 is the
initial pose; frame N is the post-step pose after N control steps. Training
errors use that step's pre-advance reference, matching the original rules.
The failure frame is included in playback. Switching **End clip on failure**
off reveals the already-recorded recovery without recomputing anything.
The benchmark table retains the original evaluation results; the separately
labelled recording failure marker comes from this trajectory starting at frame 0.
Non-finite world-model predictions are hidden, and non-finite simulator states
end the recording with an explicit truncation marker.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
