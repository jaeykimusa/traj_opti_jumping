# jump_full.py

from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.robot.robot import * 
from go2.utils.math_utils import *
# from go2.utils.io_utils import *
import casadi as ca
import matplotlib.pyplot as plt

# =================================================================
#   PRIMARY GOAL IS TO OPTIMIZE FULL BODY TRAJECTORY
# =================================================================

import os
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R

def printSize(matrix):
    print(matrix.shape)


dt = 0.02
N = 100

jump_path = os.path.join(os.path.dirname(__file__), "q_ref_forward_jump.txt")
q_ref = np.loadtxt(jump_path, delimiter=',')
# q_ref = interpolateMatrixToTargetColumns(q_ref, 1000)

# printSize(q_ref)
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
    q_ref = q_ref_new

# printSize(q_ref)




# getting the spatial and joint velocities
qd_ref = getZerosMatrix(getRowSize(q_ref), getColumnSize(q_ref))
for i in range(getColumnSize(q_ref)):
    if i == 0 or i == getColumnSize(q_ref) - 1:
            qd_ref[:, i] = np.zeros((NUM_Q))
    else:
        qd_ref[:, i] = (q_ref[:, i] - q_ref[:, i - 1]) / dt



# print(qd_ref)

plt.plot(qd_ref[0], label=f"x")
plt.plot(qd_ref[1], label=f"y")
plt.plot(qd_ref[2], label=f"z")
plt.title("CoM Spatial Acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleartion")
plt.legend()
plt.show()

plt.plot(qd_ref[3], label=f"rx")
plt.plot(qd_ref[4], label=f"ry")
plt.plot(qd_ref[5], label=f"rz")
plt.title("CoM Angular Acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleartion")
plt.legend()
plt.show()

plt.plot(qd_ref[6], label=f"FL1")
plt.plot(qd_ref[7], label=f"FL2")
plt.plot(qd_ref[8], label=f"FL3")
plt.title("FL Joint Acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleartion")
plt.legend()
plt.show()

exit()

# cost function
cost_func = 0

# decision variables
q = ca.SX.sym("q", NUM_Q)
qd = ca.SX.sym("qd", NUM_Q)
u = ca.SX.sym("u", NUM_U)

# opti init
opti = ca.Opti()

Q = opti.variable(NUM_Q, N)
V = opti.variable(NUM_Q, N)
U = opti.variable(NUM_Q, N - 1)

# set initial and final config
q_init = q_ref[:, 0]
qd_init = qd_ref[:, 0]
q_final = q_ref[:, N - 1]
qd_final = qd_ref[: N - 1]





