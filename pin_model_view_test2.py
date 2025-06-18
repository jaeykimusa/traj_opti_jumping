from pathlib import Path
from sys import argv

import pinocchio
import numpy as np
 
# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = Path(__file__).parent.parent / "traj_opti_jumping"
 
# You should change here to set up your own URDF file or just pass it as an argument of
# this example.
urdf_filename = (
    pinocchio_model_dir / "assets/models/go2/go2.urdf"
    if len(argv) < 2
    else argv[1]
)
 
# Load the urdf model
model = pinocchio.buildModelFromUrdf(urdf_filename, pinocchio.JointModelFreeFlyer())
robot = pinocchio.RobotWrapper(model)
data = model.createData()

 
# Sample a random configuration
# q = pinocchio.randomConfiguration(model)
# q = np.array([0.0, 0.0, 0.4, 0.4, 
#               0.0, 0.0, 0.0, 
#               0.0, 0.95, -1.75, 
#               0.0, 0.95, -1.75, 
#               0.0, 0.95, -1.75, 
#               0.0, 0.95, -1.75], dtype=np.double)
# print(f"q: {q.T}")

base_xyz = np.array([0.0, 0.0, 0.3]) # init guess for z = 0.3 m
base_quat = np.array([1.0, 0.0, 0.0, 0.0])
joints = np.array([0.0, 0.95, -1.75] * 4)
q = np.concatenate([base_xyz, base_quat, joints])

feet_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
feet_ids = [model.getFrameId(name) for name in feet_names]

# Desired foot positions in world frame (z=0)
target_positions = {
    "FL_calf": np.array([0.1934, 0.142, 0]),
    "FR_calf": np.array([0.1934, -0.142, 0]),
    "RL_calf": np.array([-0.1934, 0.142, 0]),
    "RR_calf": np.array([-0.1934, -0.142, 0]),
}

# ====================
# 3. Inverse Kinematics Setup
# ====================
def pose_error(q):
    pinocchio.framesForwardKinematics(model, data, q)
    error = []
    for name, fid in zip(feet_names, feet_ids):
        err = data.oMf[fid].translation - target_positions[name]
        error.append(err)
    return np.concatenate(error)

# Solve IK to achieve desired foot positions
from scipy.optimize import least_squares

result = least_squares(
    fun=pose_error,
    x0=q,
    method="trf",
    max_nfev=100,
    verbose=2,
)

q_optimized = result.x

# ====================
# 4. Adjust Base Height
# ====================
# Compute mean foot height and adjust base z
pinocchio.framesForwardKinematics(model, data, q_optimized)
foot_heights = [data.oMf[fid].translation[2] for fid in feet_ids]
base_z_adjustment = -np.mean(foot_heights)
q_optimized[2] += base_z_adjustment

# Final check
pinocchio.framesForwardKinematics(model, data, q_optimized)
print("\nFinal foot positions:")
for name, fid in zip(feet_names, feet_ids):
    pos = data.oMf[fid].translation
    print(f"{name}: {pos}")




# Use frame ID to get transformation
fl_foot_id = model.getFrameId("FL_calf")
fr_foot_id = model.getFrameId("FR_calf")
rl_foot_id = model.getFrameId("RL_calf")
rr_foot_id = model.getFrameId("RR_calf")

pinocchio.updateFramePlacement(model, data, fl_foot_id)
fl_pose = data.oMf[fl_foot_id].translation
print("FL foot position:", fl_pose)

pinocchio.updateFramePlacement(model, data, fr_foot_id)
fr_pose = data.oMf[fr_foot_id].translation
print("FR foot position:", fr_pose)

pinocchio.updateFramePlacement(model, data, rl_foot_id)
rl_pose = data.oMf[rl_foot_id].translation
print("RL foot position:", rl_pose)

pinocchio.updateFramePlacement(model, data, rr_foot_id)
rr_pose = data.oMf[rr_foot_id].translation
print("RR foot position:", rr_pose)

com = pinocchio.centerOfMass(model, data, q)
print("Center of mass:", data.com[0])





# def compute_feet_z(model, data, q, foot_frame_names):
#     pinocchio.forwardKinematics(model, data, q)
#     pinocchio.updateFramePlacements(model, data)
#     return np.array([data.oMf[model.getFrameId(name)].translation[2] for name in foot_frame_names])

# def solve_base_z(model, data, q, foot_frame_names, tol=1e-5, max_iter=20):
#     for _ in range(max_iter):
#         zs = compute_feet_z(model, data, q, foot_frame_names)
#         mean_z = np.mean(zs)
#         if abs(mean_z) < tol:
#             break
#         q[2] -= mean_z
#     return q, zs
























# v = np.zeros(model.nv)
# a = np.zeros(model.nv)

# g_grav = pinocchio.rnea(model, data, q, v, a)
# g_bl = g_grav[:6]
# g_j = g_grav[6:]

# feet_names = ["FL", "FR", "RL", "RR"]
# feet_ids = [model.getFrameId(f"{leg}_calf") for leg in feet_names]
# bl_id = model.getFrameId("base_link")
# ncontacts = len(feet_names)

# Js__feet_q = [pinocchio.computeFrameJacobian(model, data, q, fid, pinocchio.LOCAL) for fid in feet_ids]
# Js__feet_bl = [J[:3, :6] for J in Js__feet_q]

# Jc__feet_bl_T = np.vstack(Js__feet_bl).T # shapre: 5 x (3*ncontact)
# ls = np.linalg.pinv(Jc__feet_bl_T) @ g_bl  # Use matrix multiplication
# ls__f = np.split(ls, ncontacts)            # Split into per-foot forces

# pinocchio.framesForwardKinematics(model, data, q)
# ls__bl = []
# for l__f, foot_id in zip(ls__f, feet_ids):
#     l__f_fixed = np.array(l__f, dtype=np.float64).reshape(3)
#     l_sp__f = pinocchio.Force(l__f_fixed, np.zeros(3, dtype=np.float64))
#     l_sp__bl = data.oMf[bl_id].actInv(data.oMf[foot_id].act(l_sp__f))
#     ls__bl.append(l_sp__bl.vector)


# for name, force in zip(feet_names, ls__bl):
#     print(f"Contact forces at {name}: {force}")




# exit()


# vis via rerun
import rerun as rr
from mpac_logging.mpac_logging import robot_zoo
from mpac_logging.mpac_logging.rerun.robot_logger import RobotLogger
from mpac_logging.mpac_logging.rerun.utils import rerun_initialize, rerun_store

import logging

rr.init("simple_robot_example", spawn=False)
# robot_logger = RobotLogger.from_zoo("go2")
robot_logger = RobotLogger.from_zoo("go2_description")

import time

rerun_initialize("Simple test.", spawn=False)
current_time = time.time()
# robot_logger.log_initial_state(logtime=current_time)


# q = np.transpose(data.oMi)

from scipy.spatial.transform import Rotation as R

print(robot_logger.joint_names)

dt = 1.0

r = np.array([0.0, 0.0, np.sin(0), np.cos(0)])
print(r)

base_position = [q[0], q[1], q[2]]
base_orientation = r

t1, t2, t3 = q[[7, 8, 9]]
t4, t5, t6 = q[[10, 11, 12]]
t7, t8, t9 = q[[13, 14, 15]]
t10, t11, t12 = q[[16, 17, 18]]

joint_positions = {
    "FL_hip_joint" : t1,
    "FL_thigh_joint" : t2,
    "FL_calf_joint" : t3,
    "FR_hip_joint" : t4,
    "FR_thigh_joint" : t5,
    "FR_calf_joint" : t6,
    "RL_hip_joint" : t7,
    "RL_thigh_joint" : t8,
    "RL_calf_joint" : t9,
    "RR_hip_joint" : t10,
    "RR_thigh_joint" : t11,
    "RR_calf_joint" : t12,
}

robot_logger.log_state(
    logtime=current_time,
    base_position=base_position,
    base_orientation=base_orientation,
    joint_positions=joint_positions
)

current_time += dt

rr.save("go2_stand_height_estimator_test2.rrd")

exit()