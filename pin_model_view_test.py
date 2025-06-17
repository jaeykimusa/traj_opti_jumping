from pathlib import Path
from sys import argv

import pinocchio
import numpy as np
 
# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = Path(__file__).parent.parent / "ur3_experiment"
 
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

print("model name: " + model.name)
print(f"model nq: {robot.nq}")
print(f"model nv: {model.nv}")

mass = pinocchio.computeTotalMass(model)
print(f"Go2 mass: {mass} kg")

# print(f"model nu: {model.nu}")

 
# Create data required by the algorithms
data = model.createData()
 
# Sample a random configuration
# q = pinocchio.randomConfiguration(model)
q = np.array([0.0, 0.0, 0.4, 0.4, 
              0.0, 0.0, 0.0, 
              0.0, 0.95, -1.75, 
              0.0, 0.95, -1.75, 
              0.0, 0.95, -1.75, 
              0.0, 0.95, -1.75], dtype=np.double)
print(f"q: {q.T}")
 
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)
 
# q = pinocchio.neutral(model)
# v = pinocchio.utils.zero(model.nv)
# a = pinocchio.utils.zero(model.nv)

# tau = pinocchio.rnea(model, data, q, v, a)
# print(f"model nv: {tau}")

# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

for i, frame in enumerate(model.frames):
    print(f"[{i}] {frame.name} — type: {frame.type}")

# Forward kinematics must be run first
pinocchio.forwardKinematics(model, data, q)

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
print("FL foot position:", fr_pose)

pinocchio.updateFramePlacement(model, data, rl_foot_id)
rl_pose = data.oMf[rl_foot_id].translation
print("FL foot position:", rl_pose)

pinocchio.updateFramePlacement(model, data, rr_foot_id)
rr_pose = data.oMf[rr_foot_id].translation
print("FL foot position:", rr_pose)







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

rr.save("simple_robot.rrd")

exit()