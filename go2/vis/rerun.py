# rerun.py

import rerun as rr
from go2.mpac_logging.mpac_logging import robot_zoo
from go2.mpac_logging.mpac_logging.rerun.robot_logger import RobotLogger
from go2.mpac_logging.mpac_logging.rerun.utils import rerun_initialize, rerun_store

import time
from scipy.spatial.transform import Rotation as R

def visualize(vis_name, file_name=None, q=None):
    if q is None:
        raise ValueError("q must be provided to visualize robot state.")

    rr.init(vis_name, spawn=False)
    robot_logger = RobotLogger.from_zoo("go2_description")
    rerun_initialize(vis_name, spawn=True)

    current_time = time.time()

    base_position = q[:3]
    base_orientation = R.from_euler("xyz", q[3:6], degrees=False).as_quat()
    joint_positions = {
        "FL_hip_joint": q[6], 
        "FL_thigh_joint": q[7],
        "FL_calf_joint": q[8],
        "FR_hip_joint": q[9],
        "FR_thigh_joint": q[10],
        "FR_calf_joint": q[11],
        "RL_hip_joint": q[12],
        "RL_thigh_joint": q[13],
        "RL_calf_joint": q[14],
        "RR_hip_joint": q[15],
        "RR_thigh_joint": q[16],
        "RR_calf_joint": q[17],
    }

    robot_logger.log_state(
        logtime=current_time,
        base_position=base_position,
        base_orientation=base_orientation,
        joint_positions=joint_positions
    )

    if file_name is not None:
        rr.save(file_name + ".rrd")