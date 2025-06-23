# jump.py

from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.robot.robot import * 
from go2.utils.math_utils import *

# =================================================================
#   PRIMARY GOAL IS TO SOLVE IPOPT TO JUMP FORWARD
# =================================================================

import os
import scipy.sparse as sp
import casadi 
from scipy.spatial.transform import Rotation as R

def printSize(matrix):
    print(matrix.shape)

# init
q = getDefaultStandState(model, data)
pin.framesForwardKinematics(model, data, q)

qd = np.zeros(model.nv)
qdd = np.zeros(model.nv)

# READFILE: calculated reference q for 2.5 m forward jump
print(os.path.abspath(os.getcwd()))
jump_path = os.path.join(os.path.dirname(__file__), "q_ref_forward_jump.txt")
q_ref = np.loadtxt(jump_path, delimiter=',')


# exit()


# vis via rerun
import rerun as rr
from go2.mpac_logging.mpac_logging import robot_zoo
from go2.mpac_logging.mpac_logging.rerun.robot_logger import RobotLogger
from go2.mpac_logging.mpac_logging.rerun.utils import rerun_initialize, rerun_store

# robot_logger = RobotLogger.from_zoo("go2")
robot_logger = RobotLogger.from_zoo("go2_description")

import time

rerun_initialize("2.5 forward jump test", spawn=True)
current_time = time.time()
# robot_logger.log_initial_state(logtime=current_time)

dt = 0.02

for i in range(getColumnSize(q_ref)):
    q_i = q_ref[:, i]

    r = R.from_euler("y", -q_i[2], degrees=False).as_quat()  # x, y, z w
    base_position = [q_i[0], 0, q_i[1]]
    base_orientation = r

    q_FL1 = 0
    q_FL2 = -q_i[3]
    q_FL3 = -q_i[4]
    q_RL1 = 0
    q_RL2 = -q_i[5]
    q_RL3 = -q_i[6]

    q_FR1 = 0
    q_FR2 = q_FL2
    q_FR3 = q_FL3
    q_RR1 = 0
    q_RR2 = q_RL2 
    q_RR3 = q_RL3

    joint_positions = {
        "FL_hip_joint" : q_FL1, 
        "FL_thigh_joint" : q_FL2,
        "FL_calf_joint" : q_FL3,
        "FR_hip_joint" : q_FR1,
        "FR_thigh_joint" : q_FR2,
        "FR_calf_joint" : q_FR3,
        "RL_hip_joint" : q_RL1,
        "RL_thigh_joint" : q_RL2,
        "RL_calf_joint" : q_RL3,
        "RR_hip_joint" : q_RR1,
        "RR_thigh_joint" : q_RR2,
        "RR_calf_joint" : q_RR3,
    }
    robot_logger.log_state(
        logtime=current_time,
        base_position=base_position,
        base_orientation=base_orientation,
        joint_positions=joint_positions
    )
    current_time += dt

rr.save("go2_forward_jump_test.rrd")

exit()