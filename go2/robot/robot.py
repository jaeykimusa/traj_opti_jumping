from pathlib import Path
from sys import argv
from enum import Enum, auto

import pinocchio as pin
from go2.robot.morphology import *
import numpy as np

import casadi as ca
import pinocchio.casadi

# This path refers to pin source code but you can define your own directory here.
pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = (
    pin_model_dir / "go2.urdf"
    if len(argv) < 2
    else argv[1]
)
joint_model = pin.JointModelComposite(2)
joint_model.addJoint(pin.JointModelTranslation())
joint_model.addJoint(pin.JointModelSphericalZYX())
# symbolic term


print(urdf_filename)
print(joint_model)
model = pin.buildModelFromUrdf(urdf_filename, joint_model)
robot = pin.RobotWrapper(model)
data = model.createData()


# Print all frames in the model
print("All frames in the model:")
for i, frame in enumerate(model.frames):
    print(f"{i}: {frame.name} ({frame.type})")

ad_model = pinocchio.casadi.Model(model)
ad_data = ad_model.createData()


# Numerical test of forward kinematics using Pinocchio (not CasADi)

# Define the base and end-effector frame IDs
BASE_FRAME = model.getFrameId("base")
EE_FRAME_NAMES = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
EE_FRAME_IDS = [model.getFrameId(name) for name in EE_FRAME_NAMES]