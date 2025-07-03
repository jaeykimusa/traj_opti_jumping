# io_utils.py
import numpy as np
import os
from math_utils import *
from go2.robot.robot import * 

def readData(fileName):
    jump_path = os.path.join(os.path.dirname(__file__), fileName)
    q_ref = np.loadtxt(jump_path, delimiter=',')

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

    return q_ref_new


