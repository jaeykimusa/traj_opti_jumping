from pathlib import Path
from sys import argv

import pinocchio
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.robot.robot import *

q = getDefaultStandState(model, data)
# pinocchio.framesForwardKinematics(model, data, q)

# # print("Foot positions:")
# printEEPositions(model, data)

# exit()


pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)


# base_xyz = np.array([0.0, 0.0, 0.3]) # init guess for z = 0.3 m
# base_quat = np.array([1.0, 0.0, 0.0, 0.0])
# joints = np.array([0.0, 0.95, -1.75] * 4)
# q = np.concatenate([base_xyz, base_quat, joints])


# exit()

# vis via rerun
import rerun as rr
from go2.mpac_logging.mpac_logging import robot_zoo
from go2.mpac_logging.mpac_logging.rerun.robot_logger import RobotLogger
from go2.mpac_logging.mpac_logging.rerun.utils import rerun_initialize, rerun_store
import time
from scipy.spatial.transform import Rotation as R

# rr.init("simple_robot_example", spawn=True)
# robot_logger = RobotLogger.from_zoo("go2")
robot_logger = RobotLogger.from_zoo("go2_description")


rerun_initialize("stand_ik_test", spawn=True)
current_time = time.time()
dt = 0.02
# robot_logger.log_initial_state(logtime=current_time)

for i in range(100):
    q_i = q

    base_position = q_i[:3]
    # base_orientation = q_i[3,7]
    base_orientation = np.roll(q_i[3:7], -1)
    # base_orientation = np.array([0.0, 0.0, np.sin(0), np.cos(0)])

    q_FL1 = q_i[7]
    q_FL2 = q_i[8]
    q_FL3 = q_i[9]
    q_FR1 = q_i[10]
    q_FR2 = q_i[11]
    q_FR3 = q_i[12]
    q_RL1 = q_i[13]
    q_RL2 = q_i[14]
    q_RL3 = q_i[15]
    q_RR1 = q_i[16]
    q_RR2 = q_i[17]
    q_RR3 = q_i[18]

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

rr.save("go2_stand_ik_test.rrd")

exit()