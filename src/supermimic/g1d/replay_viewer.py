"""CPU-only disk replay using the original mjlab mesh/ghost renderer every frame."""

from __future__ import annotations

import copy
import json
import queue
import time
from pathlib import Path

import mujoco
import viser
from mjlab.viewer.viser.scene import MjlabViserScene

from supermimic.g1d import constants as C
from supermimic.g1d.replay import (
  FAILURE_RULES,
  FORMAT_VERSION,
  THRESHOLDS,
  ReplayCursor,
  load_clip,
)
from supermimic.g1d.robot import load_spec
from supermimic.g1d.viewer import clip_markdown


class G1DReplayViewer:
  def __init__(
    self,
    replay_dir: Path,
    visual_mesh_dir: str | None = None,
    port: int = 8080,
    host: str = "0.0.0.0",
  ):
    self.directory = replay_dir
    self.index = json.loads((replay_dir / "index.json").read_text())
    if self.index["version"] != FORMAT_VERSION:
      raise ValueError("Unsupported recording version")
    self.entries = [
      v for _, v in sorted(self.index["clips"].items(), key=lambda kv: int(kv[0]))
    ]
    if not self.entries:
      raise ValueError("No completed clips in this cache")
    self.model = load_spec(visual_mesh_dir).compile()
    names = [self.model.joint(i).name for i in range(self.model.njnt)]
    if set(names) != set(self.index["joint_names"]) or self.model.nq != len(names):
      raise ValueError("Recording's joint layout does not match this robot")
    self.qorder = [self.index["joint_names"].index(n) for n in names]
    self.data = mujoco.MjData(self.model)
    self.server = viser.ViserServer(host=host, port=port, label="G1-D offline replay")
    self.scene = MjlabViserScene(self.server, self.model, num_envs=1)
    self.scene.debug_visualization_enabled = True
    self.scene.camera_tracking_enabled = True
    self.scene._tracked_body_id = self.model.body(C.ROOT_BODY).id
    if visual_mesh_dir is None:
      self.scene.geom_groups_visible[3] = True
      self.scene._sync_visibilities()
    self.ghosts = []
    for color in ((0.1, 0.85, 0.2, 0.6), (0.2, 0.45, 1.0, 0.6)):
      model = copy.deepcopy(self.model)
      for gi in range(model.ngeom):
        model.geom_rgba[gi] = color
        if visual_mesh_dir is not None and model.geom_group[gi] != 2:
          model.geom_rgba[gi, 3] = 0
      self.ghosts.append(model)
    self.grid = self.server.scene.add_grid(
      "/floor", width=200, height=200, cell_size=0.5, section_size=5
    )
    self.commands = queue.SimpleQueue()
    self.selected = 0
    self.filtered = list(range(len(self.entries)))
    self.filter_text = ""
    self.failed_only = False
    self.paused = False
    self.speed = 1.0
    self.auto_next = False
    self.end_on_failure = True
    self.show_ref = self.show_wm = True
    self.dirty = True
    self.render_ms = 0.0
    self._build_gui()
    self.select(0)

    @self.server.on_client_connect
    def connected(client):
      client.camera.position = (2.8, -3.2, 2.0)
      client.camera.look_at = (0.0, 0.0, 0.8)
      self.commands.put(("redraw", None))

  def _build_gui(self):
    gui = self.server.gui

    def change(handle, command, transform=lambda x: x):
      @handle.on_update
      def updated(event):
        # Server-side slider/selection refreshes must not pause or seek playback.
        if event.client is not None:
          self.commands.put((command, transform(event.target.value)))

    def button(label, command, value=None):
      handle = gui.add_button(label)
      handle.on_click(lambda _: self.commands.put((command, value)))
      return handle

    with gui.add_folder("Clips"):
      self.filter = gui.add_text("Filter", initial_value="")
      self.failed = gui.add_checkbox("Failed only (train rule)", initial_value=False)
      change(self.filter, "filter_text")
      change(self.failed, "failed_only")
      self.dropdown = gui.add_dropdown(
        "Motion Clip", options=[e["name"] for e in self.entries]
      )
      change(self.dropdown, "select")
      self.count = gui.add_markdown("")
      nav = gui.add_button_group("Clip", ("◀ Prev", "Next ▶"))
      nav.on_click(
        lambda ev: self.commands.put(
          ("clip_step", -1 if ev.target.value.startswith("◀") else 1)
        )
      )
      change(
        gui.add_checkbox("Auto next (clip end or failure)", initial_value=False),
        "auto_next",
      )
      self.clip_md = gui.add_markdown("")
    with gui.add_folder("Playback"):
      self.play_button = button("Pause", "toggle_pause")
      nav = gui.add_button_group("Frame", ("◀ Back", "Step ▶"))
      nav.on_click(
        lambda ev: self.commands.put(
          ("step", -1 if ev.target.value.startswith("◀") else 1)
        )
      )
      button("Restart clip", "seek", 0)
      self.failure_button = button("Jump to first failure", "failure")
      self.slider = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
      change(self.slider, "seek", int)
      speed = gui.add_dropdown(
        "Speed", options=("0.1x", "0.25x", "0.5x", "1x", "2x"), initial_value="1x"
      )
      change(speed, "speed", lambda v: float(v[:-1]))
      change(
        gui.add_checkbox("End clip on failure", initial_value=True), "end_on_failure"
      )
      self.live_md = gui.add_markdown("")
    with gui.add_folder("Display"):
      change(
        gui.add_checkbox("Reference ghost (green)", initial_value=True), "show_ref"
      )
      change(
        gui.add_checkbox("World-model ghost (blue)", initial_value=True), "show_wm"
      )
      change(gui.add_checkbox("Follow chassis", initial_value=True), "follow")
    with gui.add_folder("Recording"):
      size = sum(e["bytes"] for e in self.entries) / 2**20
      gui.add_markdown(
        f"**Offline · {1 / self.index['dt']:.0f} Hz** · {len(self.entries)} clips · {size:.1f} MiB on disk\n\n"
        "Grey: simulator · green: reference · blue: closed-loop world model.\n\n"
        "Drag the frame slider to pause and inspect. Failures are recorded; changing the stop toggle does not rerun inference."
      )
      self.status = gui.add_markdown("")
    self.refilter()

  def refilter(self):
    needle = self.filter_text.strip().lower()
    self.filtered = [
      i
      for i, e in enumerate(self.entries)
      if needle in e["name"].lower()
      and (not self.failed_only or e["first_failure"] is not None)
    ]
    options = [self.entries[i]["name"] for i in self.filtered]
    self.dropdown.disabled = not options
    self.dropdown.options = options or ["(no matches)"]
    self.count.content = f"{len(options)} of {len(self.entries)} clips"
    if options:
      if self.selected not in self.filtered:
        self.commands.put(("select", options[0]))
      else:
        self.dropdown.value = self.entries[self.selected]["name"]

  def select(self, index):
    entry = self.entries[index]
    # Only one decompressed clip is retained; the dataset is never loaded here.
    self.arrays = load_clip(self.directory, entry, len(self.qorder))
    self.selected = index
    self.cursor = ReplayCursor(entry["frames"], entry["first_failure"])
    self.cursor.end_on_failure = self.end_on_failure
    self.slider.value = 0
    self.slider.max = max(1, self.cursor.end)
    self.slider.disabled = self.cursor.end == 0
    self.failure_button.disabled = self.cursor.failure is None
    self.dropdown.value = entry["name"]
    benchmark = self.index.get("benchmark", {}).get(entry["name"], {})
    failure = entry["first_failure"]
    note = (
      "none"
      if failure is None
      else f"frame {failure} ({', '.join(entry['fail_rules'])})"
    )
    self.clip_md.content = (
      clip_markdown(entry["name"], benchmark)
      + f"\n\n**This recording:** first failure {note}."
    )
    if entry.get("truncated"):
      self.clip_md.content += (
        "\n\nRecording stopped early: simulator produced a non-finite state."
      )
    self.dirty = True

  def clip_step(self, offset):
    if self.filtered:
      i = self.filtered.index(self.selected) if self.selected in self.filtered else 0
      self.select(self.filtered[(i + offset) % len(self.filtered)])

  def process_commands(self):
    changed = False
    while not self.commands.empty():
      command, value = self.commands.get()
      changed = True
      if command == "select":
        index = next(
          (i for i, e in enumerate(self.entries) if e["name"] == value), None
        )
        if index is not None and index in self.filtered:
          self.select(index)
      elif command in ("filter_text", "failed_only"):
        setattr(self, command, value)
        self.refilter()
      elif command == "clip_step":
        self.clip_step(value)
      elif command == "toggle_pause":
        self.paused = not self.paused
      elif command in ("seek", "step", "failure"):
        self.paused = True
        frame = (
          self.cursor.frame + value
          if command == "step"
          else self.cursor.failure
          if command == "failure"
          else value
        )
        if frame is not None:
          self.cursor.seek(frame)
      elif command == "follow":
        self.scene.camera_tracking_enabled = value
      elif command in ("speed", "auto_next", "show_ref", "show_wm"):
        setattr(self, command, value)
      elif command == "end_on_failure":
        self.end_on_failure = self.cursor.end_on_failure = value
        self.cursor.seek(self.cursor.frame)
        self.slider.value = self.cursor.frame
        self.slider.max = max(1, self.cursor.end)
        self.slider.disabled = self.cursor.end == 0
      self.dirty = True
    self.play_button.label = "Play" if self.paused else "Pause"
    return changed

  def render(self):
    frame = self.cursor.frame
    pose = self.arrays["poses"][frame][:, self.qorder]
    self.data.qpos[:] = pose[0]
    mujoco.mj_kinematics(self.model, self.data)
    self.scene.clear()
    for i, show in enumerate(
      (self.show_ref, self.show_wm and self.arrays["wm_valid"][frame])
    ):
      if show:
        self.scene.add_ghost_entity(
          pose[i + 1],
          model=self.ghosts[i],
          alpha=0.6,
          label=("reference", "wm_prediction")[i],
        )
    # Keep the original renderer and all per-frame ghost updates. Atomic delivery
    # keeps the three poses together; no decimation or stale ghost frames.
    with self.server.atomic():
      # mjviser follows the chassis by translating the scene. The world-fixed
      # floor needs the same XYZ translation, otherwise it moves with the robot
      # and makes locomotion look like motion in place.
      self.grid.position = (
        -self.data.xpos[self.model.body(C.ROOT_BODY).id]
        if self.scene.camera_tracking_enabled
        else (0.0, 0.0, 0.0)
      )
      self.scene.update_from_mjdata(self.data)
      self.slider.value = frame
    self.dirty = False

  def update_text(self, fps):
    frame = self.cursor.frame
    rows = ["| rule | now | threshold |", "|---|---|---|"]
    for name, error, threshold in zip(
      FAILURE_RULES, self.arrays["errors"][frame], THRESHOLDS, strict=True
    ):
      flag = " **over**" if error > threshold else ""
      rows.append(f"| {name} | {error * 100:.1f} cm{flag} | {threshold * 100:g} cm |")
    self.live_md.content = (
      f"frame **{frame} / {self.cursor.frames - 1}** · {frame * self.index['dt']:.2f} s\n\n"
      + "\n".join(rows)
    )
    if frame == 0:
      self.live_md.content += "\n\nInitial pose; no control step yet."
    elif not self.arrays["wm_valid"][frame]:
      self.live_md.content += "\n\nWorld-model prediction diverged; blue ghost hidden."
    size = sum(a.nbytes for a in self.arrays.values()) / 2**20
    self.status.content = f"{fps:.1f} server frames/s · render {self.render_ms:.1f} ms · current clip {size:.2f} MiB in RAM\n\nErrors use the training step's target (one frame before the displayed post-step reference)."

  def run(self):
    deadline = time.monotonic()
    stats_at, drawn = deadline, 0
    text_at, fps = deadline, 0.0
    try:
      while True:
        changed = self.process_commands()
        now = time.monotonic()
        if changed:
          deadline = now + self.index["dt"] / self.speed
        # No frame skipping: an overloaded browser/server slows playback rather
        # than omitting the moment of failure.
        if not self.paused and now >= deadline and self.server.get_clients():
          if self.cursor.advance() and self.auto_next:
            self.clip_step(1)
          self.dirty = True
          deadline = max(deadline + self.index["dt"] / self.speed, now)
        if self.dirty:
          t0 = time.monotonic()
          self.render()
          self.render_ms = (time.monotonic() - t0) * 1000
          drawn += 1
        now = time.monotonic()
        if now - stats_at >= 1.0:
          fps = drawn / (now - stats_at)
          stats_at, drawn = now, 0
        if changed or now - text_at >= 0.2:
          self.update_text(0.0 if self.paused else fps)
          text_at = now
        time.sleep(max(0.001, min(0.005, deadline - time.monotonic())))
    except KeyboardInterrupt:
      pass
    finally:
      self.server.stop()
