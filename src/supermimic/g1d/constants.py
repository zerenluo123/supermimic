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

# Chassis and column commands are MAX * tanh(a), the same in the simulator and in
# imagination: a hard clip lets the policy push the raw action far past it.
VX_MAX, WZ_MAX, VZ_MAX = 1.5, 0.6, 0.0765
# First-order time constant of the chassis / column velocity drives.
TAU = 0.15

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
