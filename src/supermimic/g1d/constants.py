"""G1-D layout, ported from SuperMimic-Reproduced (g1d).

The plant keeps the MJCF joints: three planar chassis joints (x, y slides, yaw hinge),
two lift slides coupled by an equality constraint, and 16 upper-body hinges. The
learning state collapses the slides into one column height, so a state joint
vector is ``[column, 16 upper joints]`` (17) and the full state is GenCoord's
``[root_pos(3), root_quat(4), qj(17), root_lin_vel(3), root_ang_vel(3), vj(17)]``
with ``AGV_link`` as the root.
"""

from __future__ import annotations

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "g1d"
G1D_XML = ASSET_DIR / "g1d.xml"

BASE_JOINTS = ("base_x_joint", "base_y_joint", "base_yaw_joint")
LIFT_JOINTS = ("LZ_mt_Joint", "LZ_it_Joint")
# Yaw_Joint is the waist *pitch* (axis y), torso_Joint the waist yaw.
UPPER_JOINTS = (
  "Yaw_Joint",
  "torso_Joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
COLUMN = "column"
STATE_JOINTS = (COLUMN, *UPPER_JOINTS)
NJ = len(STATE_JOINTS)  # 17
NUM_UPPER = len(UPPER_JOINTS)  # 16
ACT_DIM = NUM_UPPER + 3  # 16 position offsets + vx, wz, vz
COLUMN_LIMIT = (0.0, 0.42)

ROOT_BODY = "AGV_link"
AGV_HEIGHT = 0.1113
# Body 0 is the root and the anchor, as the reward and the reference re-anchoring expect.
TRACKED_BODIES = (
  "AGV_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
UPPER_TRACKED_BODIES = TRACKED_BODIES[1:]

# URR keypoints (Dyna-2.1 §3.1.1) as the g1d-urr controller places them: the chest
# 0.2 m above the torso_link origin, the hands at left/right_hand_palm_link (fixed
# to the wrist yaw links, g1d.xml), the elbows at the elbow links, each a tracked
# body plus an offset in its frame. The chest and the hands carry an orientation.
URR_BODIES = (
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
)
URR_OFFSETS = (
  (0.0, 0.0, 0.2),
  (0.0415, 0.003, 0.0),
  (0.0415, -0.003, 0.0),
  (0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0),
)
# The body keypoint's offset from the torso_link origin by ``body_point``: the chest
# above it (``URR_OFFSETS``), or the origin itself, the waist yaw joint, where
# ScaleBFM would put it (it uses link origins).
URR_BODY_OFFSETS = {"chest": URR_OFFSETS[0], "waist": (0.0, 0.0, 0.0)}
URR_NUM_ORIENTED = 3
URR_POSE_DIM = 3 * len(URR_BODIES) + 6 * URR_NUM_ORIENTED  # positions, rot6d: 33
URR_VEL_DIM = 3 * len(URR_BODIES) + 3 * URR_NUM_ORIENTED  # linear, angular: 24

# The parts of a robot-frame actor goal in its order: the chassis, then the URR
# keypoints by their bodies (the body point, the palms, the elbows). ScaleBFM's
# modes (its ``mode_candidates``) each give the actor some of them.
GOAL_PARTS = ("chassis", *URR_BODIES)
GOAL_MODES = {
  "full": GOAL_PARTS,
  "three_point": ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
}

# Chassis and column commands are MAX * tanh(a), the same in the simulator and in
# imagination: a hard clip lets the policy push the raw action far past it.
VX_MAX, WZ_MAX, VZ_MAX = 1.5, 0.6, 0.0765
# First-order time constant of the chassis / column velocity drives. A placeholder from
# supermimic_vibed, not a measurement.
TAU = 0.15
# Stiff chassis (G1DActionCfg.stiff_chassis): the command reaches the drives through a
# first-order filter with TAU, and the drives hold their target within STIFF_LOOP_TAU,
# up to the traction limits below. Estimates until the chassis is measured: friction
# ~0.8 * 70.7 kg * g per axis, and that force at ~0.18 m (half the wheel track) for yaw.
STIFF_LOOP_TAU = 0.005
CHASSIS_FORCE_LIMIT = 550.0  # N, each of x and y
CHASSIS_TORQUE_LIMIT = 100.0  # N·m, yaw

# Upper-body PD follows the G1 motor model (BeyondMimic gains). The waist pitch is a
# self-locking motor with the Unitree example gains; its effort is a placeholder.
UPPER_MOTORS = (
  "waist_pitch",
  "waist_yaw",
  "5020",  # shoulder pitch
  "5020",  # shoulder roll
  "5020",  # shoulder yaw
  "5020",  # elbow
  "5020",  # wrist roll
  "4010",  # wrist pitch
  "4010",  # wrist yaw
  "5020",
  "5020",
  "5020",
  "5020",
  "5020",
  "4010",
  "4010",
)
WAIST_PITCH_KP, WAIST_PITCH_KD, WAIST_PITCH_EFFORT = 500.0, 12.0, 120.0
WAIST_PITCH_SCALE = 0.5

# Arm defaults are the G1 standing pose; column and waist at 0.
DEFAULT_JOINT_POS = {
  "left_elbow_joint": 0.6,
  "right_elbow_joint": 0.6,
  "left_shoulder_roll_joint": 0.2,
  "left_shoulder_pitch_joint": 0.2,
  "right_shoulder_roll_joint": -0.2,
  "right_shoulder_pitch_joint": 0.2,
}

# Failure rules (training termination = benchmark failure, as the G1's pelvis rules):
# the chassis world xy distance to the reference, the distance of every upper
# tracked body to its reference re-anchored at the chassis (xy + yaw), and the
# height of the column top (Yaw_Link) against the reference's.
CHASSIS_THRESHOLD = 0.1
UPPER_THRESHOLD = 0.25
COLUMN_THRESHOLD = 0.1

# Proprio noise half-widths of the column entries (metres, m/s): the joint ones
# (0.01 rad, 0.5 rad/s) would drown a column that moves at most 0.0765 m/s.
COLUMN_POS_NOISE = 0.01
COLUMN_VEL_NOISE = 0.01
# Training noise on the reference chassis position the actor sees (odometry error).
ANCHOR_POS_NOISE = 0.05
# With the chassis window, the same odometry error on the robot's own pose: half-width
# of its x, y (``ANCHOR_POS_NOISE``) and yaw (rad), drawn every step.
ANCHOR_YAW_NOISE = 0.05
