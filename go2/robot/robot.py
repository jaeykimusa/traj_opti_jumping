from pathlib import Path
from sys import argv
from enum import Enum, auto

import pinocchio as pin
from go2.robot.morphology import *
import numpy as np

# # This path refers to pin source code but you can define your own directory here.
# pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
 
# # You should change here to set up your own URDF file or just pass it as an argument of this example.
# urdf_filename = (
#     pin_model_dir / "model/go2/go2.urdf"
#     if len(argv) < 2
#     else argv[1]
# )

# joint_model = pin.JointModelComposite(2)
# joint_model.addJoint(pin.JointModelTranslation())
# joint_model.addJoint(pin.JointModelSphericalZYX())

# # Load the urdf model
# model = pin.buildModelFromUrdf(urdf_filename, pin.JointModelFreeFlyer())
# geom_model = pin.buildGeomFromUrdf(model, urdf_filename, pin.GeometryType.COLLISION)
# robot = pin.RobotWrapper(model)
# data = model.createData()
# geom_data = geom_model.createData()

# # symbolic term
# from pinocchio.casadi import Model as CasadiModel
# ad_model = CasadiModel(model)  
# ad_data = ad_model.createData()

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
model = pin.buildModelFromUrdf(urdf_filename, joint_model)
robot = pin.RobotWrapper(model)
data = model.createData()

ad_model = pinocchio.casadi.Model(model)
ad_data = ad_model.createData()

# for i, frame in enumerate(model.frames):
#     frameType = frame.type.name if hasattr(frame.type, 'name') else str(frame.type)
#     print(f"{frame.name}, {i}, {frameType}")


# pin.forwardKinematics(model, data, q)
# pin.updateFramePlacements(model, data)

# FL_thigh_pos = data.oMf[Frame.FL_thigh].translation
# FL_calf_pos = data.oMf[Frame.FL_calf].translation
# FL_thigh_length = np.linalg.norm(FL_calf_pos - FL_thigh_pos)
# print(f"Thigh_length: {FL_thigh_length:.5f} m")

# FL_EE_pos = data.oMf[Frame.FL_EE].translation
# print(FL_calf_pos)
# print(FL_EE_pos)
# FL_calf_length = np.linalg.norm(FL_thigh_pos - FL_EE_pos)
# print(f"Calf_length: {FL_calf_length:.5f} m")

# RL_thigh_pos = data.oMf[Frame.RL_thigh].translation
# L_body_length = np.linalg.norm(RL_thigh_pos - FL_thigh_pos)
# print(f"Body_length 1: {L_body_length:.5f} m")

# FR_thigh_pos = data.oMf[Frame.FR_thigh].translation
# RR_thigh_pos = data.oMf[Frame.RR_thigh].translation
# R_body_length = np.linalg.norm(RR_thigh_pos - FR_thigh_pos)
# print(f"Body_length 2: {R_body_length:.5f} m")
# print(2 * 0.1934)



