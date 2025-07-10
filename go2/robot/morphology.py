# morphology.py

from enum import IntEnum, Enum, auto
# from robot import *

class Frame(IntEnum):
    UNIVERSE = 0        # type FIXED_JOINT
    ROOT_JOINT = 1      # type JOINT
    BASE_LINK = 2       # type BODY
    FL_hip_joint = 3    # type JOINT
    FL_hip = 4
    FL_thigh_joint = 5
    FL_thigh = 6
    FL_calf_joint = 7
    FL_calf = 8
    FL_EE_joint = 9
    FL_EE = 10
    FR_hip_joint = 11
    FR_hip = 12
    FR_thigh_joint = 13
    FR_thigh = 14
    FR_calf_joint = 15
    FR_calf = 16
    FR_EE_joint = 17
    FR_EE = 18
    RL_hip_joint = 19
    RL_hip = 20
    RL_thigh_joint = 21
    RL_thigh = 22
    RL_calf_joint = 23
    RL_calf = 24
    RL_EE_joint = 25
    RL_EE = 26
    RR_hip_joint = 27
    RR_hip = 28
    RR_thigh_joint = 29
    RR_thigh = 30 
    RR_calf_joint = 31
    RR_calf = 32
    RR_EE_joint = 33
    RR_EE =34

NUM_Q = 18
NUM_U = 12
NUM_F = 12
NUM_C = 4

C_FRAME_IDS = [Frame.FL_EE, Frame.FR_EE, Frame.RL_EE, Frame.RR_EE]

# [0] universe — type: FIXED_JOINT
# [1] root_joint — type: JOINT
# [2] base_link — type: BODY
# [3] FL_hip_joint — type: JOINT
# [4] FL_hip — type: BODY
# [5] FL_thigh_joint — type: JOINT
# [6] FL_thigh — type: BODY
# [7] FL_calf_joint — type: JOINT
# [8] FL_calf — type: BODY
# [9] FL_EE_joint — type: FIXED_JOINT
# [10] FL_EE — type: BODY
# [11] FR_hip_joint — type: JOINT
# [12] FR_hip — type: BODY
# [13] FR_thigh_joint — type: JOINT
# [14] FR_thigh — type: BODY
# [15] FR_calf_joint — type: JOINT
# [16] FR_calf — type: BODY
# [17] FR_EE_joint — type: FIXED_JOINT
# [18] FR_EE — type: BODY
# [19] RL_hip_joint — type: JOINT
# [20] RL_hip — type: BODY
# [21] RL_thigh_joint — type: JOINT
# [22] RL_thigh — type: BODY
# [23] RL_calf_joint — type: JOINT
# [24] RL_calf — type: BODY
# [25] RL_EE_joint — type: FIXED_JOINT
# [26] RL_EE — type: BODY
# [27] RR_hip_joint — type: JOINT
# [28] RR_hip — type: BODY
# [29] RR_thigh_joint — type: JOINT
# [30] RR_thigh — type: BODY
# [31] RR_calf_joint — type: JOINT
# [32] RR_calf — type: BODY
# [33] RR_EE_joint — type: FIXED_JOINT
# [34] RR_EE — type: BODY