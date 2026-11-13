"""Play an offline G1-D recording on the CPU, with full visual meshes if supplied."""

from pathlib import Path

import tyro


def main(
  replay_dir: Path,
  visual_mesh_dir: str | None = None,
  port: int = 8080,
  host: str = "0.0.0.0",
) -> None:
  from supermimic.g1d.replay_viewer import G1DReplayViewer

  G1DReplayViewer(replay_dir, visual_mesh_dir, port, host).run()


if __name__ == "__main__":
  tyro.cli(main)
