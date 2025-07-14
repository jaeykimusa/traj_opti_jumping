# io_utils.py
import numpy as np
import os
from go2.utils.math_utils import *
from go2.robot.robot import * 

def readReferenceData(fileNameQ, fileNameQD, fileNameF, basePath=None):
    if basePath is None:
        basePath = os.path.dirname(__file__)
    jump_path = os.path.join(basePath, fileNameQ)
    q_ref = np.loadtxt(jump_path, delimiter=',')
    jump_path = os.path.join(basePath, fileNameQD)
    qd_ref = np.loadtxt(jump_path, delimiter=',')
    jump_path = os.path.join(basePath, fileNameF)
    f_ref = np.loadtxt(jump_path, delimiter=',')

    # Q[0] = x
    # Q[1] = y
    # Q[2] = z
    # Q[3] = rx
    # Q[4] = ry
    # Q[5] = rz
    # Q[6] = FL1
    # Q[7] = FL2
    # Q[8] = FL3
    # Q[9] = FR1
    # Q[10] = FR2
    # Q[11] = FR3
    # Q[12] = RL1
    # Q[13] = RL2
    # Q[14] = RL3
    # Q[15] = RR1
    # Q[16] = RR2
    # Q[17] = RR3

    # F[0] = FL_X
    # F[1] = FL_Y
    # F[2] = FL_Z
    # F[3] = FR_X
    # F[4] = FR_Y
    # F[5] = FR_Z
    # F[6] = RL_X
    # F[7] = RL_Y
    # F[8] = RL_Z
    # F[9] = RR_X
    # F[10] = RR_Y
    # F[11] = RR_Z

    if (getRowSize(q_ref) == 7):
        q_ref_new = getZerosMatrix(NUM_Q, getColumnSize(q_ref))
        q_ref_new[0, :] = q_ref[0, :]
        q_ref_new[2, :] = q_ref[1, :]
        q_ref_new[4, :] = q_ref[2, :]
        q_ref_new[7, :] = q_ref[3, :] # FL2
        q_ref_new[8, :] = q_ref[4, :] # FL3
        q_ref_new[10, :] = q_ref[3, :] # FR2
        q_ref_new[11, :] = q_ref[4, :] # FR3
        q_ref_new[13, :] = q_ref[5, :] # RL2
        q_ref_new[14, :] = q_ref[6, :] # RL3
        q_ref_new[16, :] = q_ref[5, :] # RR2
        q_ref_new[17, :] = q_ref[6, :] # RR3

    if (getRowSize(qd_ref) == 3 and getRowSize(f_ref) == 8):
        qd_ref_new = getZerosMatrix(NUM_Q, getColumnSize(q_ref))
        f_ref_new = getZerosMatrix(NUM_F, getColumnSize(f_ref) + 1)
        qd_ref_new[0, :] = qd_ref[0, :] # x
        qd_ref_new[2, :] = qd_ref[1, :] # z
        qd_ref_new[4, :] = qd_ref[2, :] # ry
        # qd_ref_new[7, :] = f_ref[4, :] + f_ref[4, -1] / 0.2 # FL2
        # qd_ref_new[8, :] = f_ref[5, :] + f_ref[5, -1] / 0.2 # FL3
        # qd_ref_new[10, :] = f_ref[4, :] + f_ref[4, -1] / 0.2 # FR2
        # qd_ref_new[11, :] = f_ref[5, :] + f_ref[5, -1] / 0.2 # FR3
        # qd_ref_new[13, :] = f_ref[6, :] + f_ref[6, -1] / 0.2 # RL2
        # qd_ref_new[14, :] = f_ref[7, :] + f_ref[7, -1] / 0.2 # RL3
        # qd_ref_new[16, :] = f_ref[6, :] + f_ref[6, -1] / 0.2 # RR2
        # qd_ref_new[17, :] = f_ref[7, :] + f_ref[7, -1] / 0.2 # RR3
        # f_ref_new[0, :] = f_ref[0, :] + f_ref[0, -1] / 0.2 # FL_X
        # f_ref_new[2, :] = f_ref[1, :] + f_ref[1, -1] / 0.2 # FL_Z
        # f_ref_new[3, :] = f_ref[0, :] + f_ref[0, -1] / 0.2 # FR_X
        # f_ref_new[5, :] = f_ref[1, :] + f_ref[1, -1] / 0.2 # FR_Z
        # f_ref_new[6, :] = f_ref[2, :] + f_ref[2, -1] / 0.2 # RL_X
        # f_ref_new[8, :] = f_ref[3, :] + f_ref[3, -1] / 0.2 # RL_Z
        # f_ref_new[9, :] = f_ref[2, :] + f_ref[2, -1] / 0.2 # RR_X
        # f_ref_new[11, :] = f_ref[3, :] + f_ref[3, -1] / 0.2 # RR_Z

        qd_ref_new[7, :] = np.hstack([f_ref[4, :], f_ref[4, -1]]) #/ 0.2]) # FL2
        qd_ref_new[8, :] = np.hstack([f_ref[5, :], f_ref[5, -1]]) # FL3
        qd_ref_new[10, :] = np.hstack([f_ref[4, :], f_ref[4, -1]]) # FR2
        qd_ref_new[11, :] = np.hstack([f_ref[5, :], f_ref[5, -1]]) # FR3
        qd_ref_new[13, :] = np.hstack([f_ref[6, :], f_ref[6, -1]]) # RL2
        qd_ref_new[14, :] = np.hstack([f_ref[7, :], f_ref[7, -1]]) # RL3
        qd_ref_new[16, :] = np.hstack([f_ref[6, :], f_ref[6, -1]]) # RR2
        qd_ref_new[17, :] = np.hstack([f_ref[7, :], f_ref[7, -1]]) # RR3
        f_ref_new[0, :] = np.hstack([f_ref[0, :], f_ref[0, -1]]) # FL_X
        f_ref_new[2, :] = np.hstack([f_ref[1, :], f_ref[1, -1]]) # FL_Z
        f_ref_new[3, :] = np.hstack([f_ref[0, :], f_ref[0, -1]]) # FR_X
        f_ref_new[5, :] = np.hstack([f_ref[1, :], f_ref[1, -1]]) # FR_Z
        f_ref_new[6, :] = np.hstack([f_ref[2, :], f_ref[2, -1]]) # RL_X
        f_ref_new[8, :] = np.hstack([f_ref[3, :], f_ref[3, -1]]) # RL_Z
        f_ref_new[9, :] = np.hstack([f_ref[2, :], f_ref[2, -1]]) # RR_X
        f_ref_new[11, :] = np.hstack([f_ref[3, :], f_ref[3, -1]]) # RR_Z

    return q_ref_new, qd_ref_new, f_ref_new


