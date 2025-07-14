# morphology.py

from enum import IntEnum, Enum, auto
# from robot import *

MU = 0.8
INFINITY = float('inf')

# class Frame(IntEnum):
#     UNIVERSE = 0        # type FIXED_JOINT
#     ROOT_JOINT = 1      # type JOINT
#     BASE_LINK = 2       # type BODY
#     FL_hip_joint = 3    # type JOINT
#     FL_hip = 4
#     FL_thigh_joint = 5
#     FL_thigh = 6
#     FL_calf_joint = 7
#     FL_calf = 8
#     FL_EE_joint = 9
#     FL_EE = 10
#     FR_hip_joint = 11
#     FR_hip = 12
#     FR_thigh_joint = 13
#     FR_thigh = 14
#     FR_calf_joint = 15
#     FR_calf = 16
#     FR_EE_joint = 17
#     FR_EE = 18
#     RL_hip_joint = 19
#     RL_hip = 20
#     RL_thigh_joint = 21
#     RL_thigh = 22
#     RL_calf_joint = 23
#     RL_calf = 24
#     RL_EE_joint = 25
#     RL_EE = 26
#     RR_hip_joint = 27
#     RR_hip = 28
#     RR_thigh_joint = 29
#     RR_thigh = 30 
#     RR_calf_joint = 31
#     RR_calf = 32
#     RR_EE_joint = 33
#     RR_EE = 34

NUM_Q = 18
NUM_U = 12
NUM_F = 12
NUM_C = 4

# EE_FRAME_IDS = [Frame.FL_EE, Frame.FR_EE, Frame.RL_EE, Frame.RR_EE] # or C_FRAME_IDS
# EE_FRAME_NAMES = [Frame.FL_EE.name, Frame.FR_EE.name, Frame.RL_EE.name, Frame.RR_EE.name]
# BODY_NAME = "BODY"
# BODY_FRAME = Frame.BASE_LINK

EE_FRAME_IDS = [26, 58, 34, 66]
FOOT_NAMES = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
BASE_FRAME = 2
BASE_NAME = "BASE"

# universe, 0, FIXED_JOINT
# root_joint, 1, JOINT
# base, 2, BODY
# FL_hip_rotor_joint, 3, FIXED_JOINT
# FL_hip_rotor, 4, BODY
# FR_hip_rotor_joint, 5, FIXED_JOINT
# FR_hip_rotor, 6, BODY
# Head_upper_joint, 7, FIXED_JOINT
# Head_upper, 8, BODY
# Head_lower_joint, 9, FIXED_JOINT
# Head_lower, 10, BODY
# LF_HAA, 11, JOINT
# FL_hip, 12, BODY
# FL_thigh_rotor_joint, 13, FIXED_JOINT
# FL_thigh_rotor, 14, BODY
# LF_HFE, 15, JOINT
# FL_thigh, 16, BODY
# FL_calf_rotor_joint, 17, FIXED_JOINT
# FL_calf_rotor, 18, BODY
# LF_KFE, 19, JOINT
# FL_calf, 20, BODY
# FL_calflower_joint, 21, FIXED_JOINT
# FL_calflower, 22, BODY
# FL_calflower1_joint, 23, FIXED_JOINT
# FL_calflower1, 24, BODY
# LF_FOOT_joint, 25, FIXED_JOINT
# LF_FOOT, 26, BODY
# LH_HAA, 27, JOINT
# RL_hip, 28, BODY
# LH_HFE, 29, JOINT
# RL_thigh, 30, BODY
# LH_KFE, 31, JOINT
# RL_calf, 32, BODY
# LH_FOOT_joint, 33, FIXED_JOINT
# LH_FOOT, 34, BODY
# RL_calflower_joint, 35, FIXED_JOINT
# RL_calflower, 36, BODY
# RL_calflower1_joint, 37, FIXED_JOINT
# RL_calflower1, 38, BODY
# RL_calf_rotor_joint, 39, FIXED_JOINT
# RL_calf_rotor, 40, BODY
# RL_thigh_rotor_joint, 41, FIXED_JOINT
# RL_thigh_rotor, 42, BODY
# RF_HAA, 43, JOINT
# FR_hip, 44, BODY
# FR_thigh_rotor_joint, 45, FIXED_JOINT
# FR_thigh_rotor, 46, BODY
# RF_HFE, 47, JOINT
# FR_thigh, 48, BODY
# FR_calf_rotor_joint, 49, FIXED_JOINT
# FR_calf_rotor, 50, BODY
# RF_KFE, 51, JOINT
# FR_calf, 52, BODY
# FR_calflower_joint, 53, FIXED_JOINT
# FR_calflower, 54, BODY
# FR_calflower1_joint, 55, FIXED_JOINT
# FR_calflower1, 56, BODY
# RF_FOOT_joint, 57, FIXED_JOINT
# RF_FOOT, 58, BODY
# RH_HAA, 59, JOINT
# RR_hip, 60, BODY
# RH_HFE, 61, JOINT
# RR_thigh, 62, BODY
# RH_KFE, 63, JOINT
# RR_calf, 64, BODY
# RH_FOOT_joint, 65, FIXED_JOINT
# RH_FOOT, 66, BODY
# RR_calflower_joint, 67, FIXED_JOINT
# RR_calflower, 68, BODY
# RR_calflower1_joint, 69, FIXED_JOINT
# RR_calflower1, 70, BODY
# RR_calf_rotor_joint, 71, FIXED_JOINT
# RR_calf_rotor, 72, BODY
# RR_thigh_rotor_joint, 73, FIXED_JOINT
# RR_thigh_rotor, 74, BODY
# RL_hip_rotor_joint, 75, FIXED_JOINT
# RL_hip_rotor, 76, BODY
# RR_hip_rotor_joint, 77, FIXED_JOINT
# RR_hip_rotor, 78, BODY
# front_camera_joint, 79, FIXED_JOINT
# front_camera, 80, BODY
# imu_joint, 81, FIXED_JOINT
# imu, 82, BODY
# radar_joint, 83, FIXED_JOINT
# radar, 84, BODY