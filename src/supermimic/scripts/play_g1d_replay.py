"""Play offline G1-D recordings on the CPU, with full visual meshes if supplied.

Several recordings are listed together, e.g. ``--replay-dir train test --label
"Train set" "Test set"`` names their clips ``[Train set] ...`` and ``[Test set] ...``.
"""

from pathlib import Path

import tyro


def main(
  replay_dir: tuple[Path, ...],
  label: tuple[str, ...] = (),
  visual_mesh_dir: str | None = None,
  port: int = 8080,
  host: str = "0.0.0.0",
) -> None:
  from supermimic.g1d.replay_viewer import G1DReplayViewer

  G1DReplayViewer(replay_dir, visual_mesh_dir, port, host, labels=label).run()


if __name__ == "__main__":
  tyro.cli(main)
