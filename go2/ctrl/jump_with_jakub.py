# jump_with_jakub.py

from go2.kinematics.kinematics import *
from go2.kinematics.fk import *
from go2.dynamics.id import *
from go2.utils.math_utils import *
from go2.robot.robot import *
from go2.robot.morphology import *
from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import scipy.sparse as sp
import os

q = getDefaultStandState(model, data)
jump_path = os.path.join(os.path.dirname(__file__), "q_ref_forward_jump.txt")
q_ref = np.loadtxt(jump_path, delimiter=',')
if (getRowSize(q_ref) == 7):
    q_ref_new = getZerosMatrix(NUM_Q, getColumnSize(q_ref))
    q_ref_new[0, :] = q_ref[0, :]
    q_ref_new[2, :] = q_ref[1, :]
    q_ref_new[4, :] = q_ref[2, :]
    q_ref_new[7, :] = -q_ref[3, :] # FL2
    q_ref_new[8, :] = -q_ref[4, :] # FL3
    q_ref_new[10, :] = -q_ref[3, :] # FR2
    q_ref_new[11, :] = -q_ref[4, :] # FR3
    q_ref_new[13, :] = -q_ref[5, :] # RL2
    q_ref_new[14, :] = -q_ref[6, :] # RL3
    q_ref_new[16, :] = -q_ref[5, :] # RR2
    q_ref_new[17, :] = -q_ref[6, :] # RR3
    q_ref = q_ref_new

plt.plot(q, label="from ik")
plt.plot(q_ref[:,0], label="from ref")
plt.title(f"q0 comparison")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.grid(True)
plt.show()