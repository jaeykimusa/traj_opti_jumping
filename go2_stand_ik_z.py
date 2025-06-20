from pathlib import Path
from sys import argv

import pinocchio
import numpy as np

from kinematics import *
from dynamics import *
from robot import *

q = getDefaultStandState(model, data)
pinocchio.framesForwardKinematics(model, data, q)

# print("Foot positions:")
printEEPositions(model, data)

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

# r = np.array([0.0, 0.0, np.sin(0), np.cos(0)])
# print(r)

base_position = [q[0], q[1], q[2]]
base_orientation = [q[3], q[4], q[5], q[6]]

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