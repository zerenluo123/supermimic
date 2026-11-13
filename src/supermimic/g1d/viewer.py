"""G1-D playback panel on top of :class:`SuperMimicViserViewer`.

Pick a clip and the policy tracks it live in the simulator, like the benchmark:
from the clip's first frame (or a random one), restarting the same clip when it
ends or fails. Added to the SuperMimic viewer: a clip filter (by name, or only
the clips that failed in a benchmark), the selected clip's benchmark results,
the live failure-rule errors, and whether a failure ends the clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from supermimic.g1d import constants as C
from supermimic.g1d.env_cfg import FAILURE_RULES, G1DMotionCommand
from supermimic.scripts.viewers import SuperMimicViserViewer

RULERS = ("train", "public")
NO_FAILURE = 1.0e9  # a failure-rule threshold nothing reaches


def load_clip_table(eval_dir: Path, tag: str) -> dict[str, dict[str, dict]]:
  """``{clip name: {ruler: clip result}}`` from ``eval_<tag>_<ruler>.json`` (the
  benchmark's per-clip results of one checkpoint)."""
  table: dict[str, dict[str, dict]] = {}
  for ruler in RULERS:
    path = eval_dir / f"eval_{tag}_{ruler}.json"
    if not path.exists():
      continue
    for clip in json.loads(path.read_text())["clips"]:
      table.setdefault(clip["name"], {})[ruler] = clip
  return table


def clip_markdown(name: str, results: dict[str, dict]) -> str:
  if not results:
    return f"**{name}**  \nno benchmark result"
  frames = next(iter(results.values()))["frames"]
  rows = [
    "| | " + " | ".join(results) + " |",
    "|---|" + "---|" * len(results),
    "| score | "
    + " | ".join(f"{r['norm_return']:.3f}" for r in results.values())
    + " |",
  ]
  failed = []
  for r in results.values():
    failed.append(
      f"step {r['steps']} ({', '.join(r['fail_rules'])})" if r["failed"] else "no"
    )
  rows.append("| failed | " + " | ".join(failed) + " |")
  for key, label, scale, unit in (
    ("xy_err_mean", "chassis xy", 100, "cm"),
    ("upper_err_mean", "upper body", 100, "cm"),
    ("col_err_mean", "column", 100, "cm"),
    ("joint_err_mean", "joints", 1, "rad"),
  ):
    rows.append(
      f"| {label} | "
      + " | ".join(f"{scale * r[key]:.2f} {unit}" for r in results.values())
      + " |"
    )
  return f"**{name}** ({frames} frames)\n\n" + "\n".join(rows)


class G1DViserViewer(SuperMimicViserViewer):
  def __init__(
    self, *args, clip_table: dict[str, dict[str, dict]] | None = None, **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.clip_table = clip_table or {}
    self._start_at_first_frame = True
    self._live_md = None
    self._live_counter = 0
    self._clip_dropdown = None
    self._auto_next = False
    self._last_clip_step: int | None = None
    terminations = self.env.unwrapped.termination_manager
    self._thresholds = {
      rule: terminations.get_term_cfg(rule).params["threshold"]
      for rule in FAILURE_RULES
      if rule in terminations.active_terms
    }

  def setup(self) -> None:
    super().setup()
    # The camera follows the first moving body, which on a fixed-base entity is
    # its still mocap wrapper: follow the chassis instead.
    mj_model = self.env.unwrapped.sim.mj_model
    self._scene._tracked_body_id = mj_model.body(f"robot/{C.ROOT_BODY}").id

  # ------------------------------------------------------------------ widgets

  def _setup_motion_clip_widget(self) -> None:
    if not self._supports_clip_selection():
      return
    gui = self._server.gui
    with gui.add_folder("Clips", order=0):
      text_filter = gui.add_text("Filter", initial_value="")
      failed_only = gui.add_checkbox(
        "Failed only (train rule)", initial_value=False, disabled=not self.clip_table
      )
      clip_dropdown = gui.add_dropdown(
        "Motion Clip", options=self.motion_names, initial_value=self.motion_names[0]
      )
      count = gui.add_markdown(f"{len(self.motion_names)} clips")
      nav = gui.add_button_group("Clip", ("◀ Prev", "Next ▶"))
      auto_next = gui.add_checkbox(
        "Auto next (clip end or failure)", initial_value=False
      )
      first_frame = gui.add_checkbox("Start at first frame", initial_value=True)
      end_on_failure = gui.add_checkbox("End clip on failure", initial_value=True)
      wm_ghost = gui.add_checkbox("World-model ghost (blue)", initial_value=True)
      self._clip_md = gui.add_markdown(self._clip_text())
      self._live_md = gui.add_markdown("")

    def refilter(_event) -> None:
      needle = text_filter.value.strip().lower()
      names = [n for n in self.motion_names if needle in n.lower()]
      if failed_only.value:
        names = [
          n for n in names if self.clip_table.get(n, {}).get("train", {}).get("failed")
        ]
      count.content = f"{len(names)} of {len(self.motion_names)} clips"
      if names:
        clip_dropdown.options = names
        if clip_dropdown.value not in names:
          clip_dropdown.value = names[0]

    text_filter.on_update(refilter)
    failed_only.on_update(refilter)
    self._clip_dropdown = clip_dropdown

    @clip_dropdown.on_update
    def _(event) -> None:
      self._selected_clip_idx = self.motion_names.index(event.target.value)
      self._clip_md.content = self._clip_text()
      self._last_clip_step = None
      self.request_reset()

    @nav.on_click
    def _(event) -> None:
      self._step_clip(-1 if event.target.value.startswith("◀") else 1)

    @auto_next.on_update
    def _(event) -> None:
      self._auto_next = bool(event.target.value)

    @wm_ghost.on_update
    def _(event) -> None:
      self.wm_prediction_enabled = bool(event.target.value)

    @first_frame.on_update
    def _(event) -> None:
      self._start_at_first_frame = bool(event.target.value)

    @end_on_failure.on_update
    def _(event) -> None:
      terminations = self.env.unwrapped.termination_manager
      for rule, threshold in self._thresholds.items():
        terminations.get_term_cfg(rule).params["threshold"] = (
          threshold if event.target.value else NO_FAILURE
        )

  def _step_clip(self, offset: int) -> None:
    """Select the clip ``offset`` places on in the filtered list (wrapping)."""
    dropdown = self._clip_dropdown
    names = list(dropdown.options)
    i = names.index(dropdown.value) if dropdown.value in names else -1
    dropdown.value = names[(i + offset) % len(names)]

  def _clip_text(self) -> str:
    name = self.motion_names[self._selected_clip_idx]
    return clip_markdown(name, self.clip_table.get(name, {}))

  # ------------------------------------------------------------------ playback

  def _apply_selected_clip(self) -> None:
    """Start the selected clip at its first frame (or a random one) and make the
    env's own resets restart it."""
    motion_cmd: G1DMotionCommand = self.env.unwrapped.command_manager.get_term("motion")  # type: ignore[assignment]
    ends = motion_cmd.motion.motion_ends
    idx = self._selected_clip_idx
    start = 0 if idx == 0 else int(ends[idx - 1])
    if self._start_at_first_frame:
      frame = start
    else:
      frame = int(torch.randint(start, int(ends[idx]) - 1, (1,)))
    env_ids = torch.zeros(1, dtype=torch.long, device=self.env.device)
    motion_cmd.forced_frames = torch.full_like(env_ids, start)
    motion_cmd.reset_to_frame(env_ids, frame)

    base_env = self.env.unwrapped
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.sim.sense()
    base_env.observation_manager.compute(update_history=True)

  def step_wm_visualizer(self, *, force: bool = False) -> bool:
    stepped = super().step_wm_visualizer(force=force)
    if stepped and self._clip_dropdown is not None:
      # The clip restarted on its own (ended or failed): with auto next, move on.
      step = self._clip_step()
      if (
        self._auto_next
        and self._last_clip_step is not None
        and step < self._last_clip_step
      ):
        self._last_clip_step = None
        self._step_clip(1)
      else:
        self._last_clip_step = step
    if stepped and self._live_md is not None:
      self._live_counter += 1
      if self._live_counter % 5 == 0:
        self._live_md.content = self._live_text()
    return stepped

  def _clip_step(self) -> int:
    """The current frame within the selected clip."""
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    idx = self._selected_clip_idx
    start = 0 if idx == 0 else int(motion_cmd.motion.motion_ends[idx - 1])
    return int(motion_cmd.time_steps[0]) - start

  def _live_text(self) -> str:
    env = self.env.unwrapped
    cache = env.__dict__.get("_g1d_step_terms")
    if cache is None:
      return ""
    terms = cache[1]
    motion_cmd = env.command_manager.get_term("motion")
    ends = motion_cmd.motion.motion_ends
    idx = self._selected_clip_idx
    start = 0 if idx == 0 else int(ends[idx - 1])
    step, length = int(motion_cmd.time_steps[0]) - start, int(ends[idx]) - start
    errors = terms["failure"][0].tolist()
    rows = ["| rule | now | threshold |", "|---|---|---|"]
    for rule, err in zip(FAILURE_RULES, errors, strict=True):
      thr = self._thresholds.get(rule)
      flag = " **over**" if thr is not None and err > thr else ""
      rows.append(
        f"| {rule} | {100 * err:.1f} cm{flag} | "
        + (f"{100 * thr:.0f} cm" if thr is not None else "off")
        + " |"
      )
    return f"live: frame {step} / {length}\n\n" + "\n".join(rows)
