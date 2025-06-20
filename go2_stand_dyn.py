# go2_stand_dyn.py

from pathlib import Path
from sys import argv

import pinocchio
import numpy as np

from kinematics import *
from dynamics import *
from robot import * 

# =================================================================
#   PRIMARY GOAL IS TO COMPUTE GROUND REACTION FORCE AT EACH FOOT
# =================================================================

# init
q = getDefaultStandState(model, data)
pinocchio.framesForwardKinematics(model, data, q)

v = np.zeros(model.nv)
a = np.zeros(model.nv)

g_grav = pinocchio.rnea(model, data, q, v, a)
g_bl = g_grav[:6]
g_j = g_grav[6:]

feet_names = ["FL", "FR", "RL", "RR"]
feet_ids = [model.getFrameId(f"{leg}_ee") for leg in feet_names]
bl_id = model.getFrameId("base_link")
ncontacts = len(feet_names)

Js__feet_q = [pinocchio.computeFrameJacobian(model, data, q, fid, pinocchio.LOCAL_WORLD_ALIGNED) for fid in feet_ids]
print("he")
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






exit()

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