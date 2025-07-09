# jump_full.py

from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.dynamics.fd import *
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


# q = ca.SX.sym("q", NUM_Q) 
# model_casadi = Model(model)
# data_casadi = model_casadi.createData()

# H = crba(model_casadi, data_casadi, convert_3Drot_to_quat(q))
# printSize(H)

# exit()

dt = 0.02
N = 100

jump_path = os.path.join(os.path.dirname(__file__), "q_ref_forward_jump.txt")
q_ref = np.loadtxt(jump_path, delimiter=',')
# q_ref = interpolateMatrixToTargetColumns(q_ref, 1000)
jump_path = os.path.join(os.path.dirname(__file__), "v_ref.txt")
v_ref = np.loadtxt(jump_path, delimiter=',')
jump_path = os.path.join(os.path.dirname(__file__), "u_ref.txt")
u_ref = np.loadtxt(jump_path, delimiter=',')

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


robot_logger.log_state(q_ref[:, 0])



# getting the spatial and joint velocities -- maybe this won't work
qd_ref = getZerosMatrix(getRowSize(q_ref), getColumnSize(q_ref))
for i in range(getColumnSize(q_ref)):
    if i == 0 or i == getColumnSize(q_ref) - 1:
            qd_ref[:, i] = np.zeros((NUM_Q))
    else:
        qd_ref[:, i] = (q_ref[:, i] - q_ref[:, i - 1]) / dt



# print(qd_ref)

# plt.plot(qd_ref[0], label=f"x")
# plt.plot(qd_ref[1], label=f"y")
# plt.plot(qd_ref[2], label=f"z")
# plt.title("CoM Spatial Acceleration")
# plt.xlabel("Time")
# plt.ylabel("Acceleartion")
# plt.legend()
# plt.show()

# plt.plot(qd_ref[3], label=f"rx")
# plt.plot(qd_ref[4], label=f"ry")
# plt.plot(qd_ref[5], label=f"rz")
# plt.title("CoM Angular Acceleration")
# plt.xlabel("Time")
# plt.ylabel("Acceleartion")
# plt.legend()
# plt.show()

# plt.plot(qd_ref[6], label=f"FL1")
# plt.plot(qd_ref[7], label=f"FL2")
# plt.plot(qd_ref[8], label=f"FL3")
# plt.title("FL Joint Acceleration")
# plt.xlabel("Time")
# plt.ylabel("Acceleartion")
# plt.legend()
# plt.show()

# exit()

# cost function
cost_func = 0

# decision variables
q = ca.SX.sym("q", NUM_Q)
qd = ca.SX.sym("qd", NUM_Q)
qdd = ca.SX.sym("qdd", NUM_Q)
u = ca.SX.sym("u", NUM_U)
f = ca.SX.sym("f", NUM_F)

# opti init
opti = ca.Opti()

# n = q_ref.shape[1]

Q = opti.variable(NUM_Q, N)
V = opti.variable(NUM_Q, N)
U = opti.variable(NUM_U, N - 1)
F = opti.variable(NUM_F, N)

# set initial and final config
q_init = q_ref[:, 0]
qd_init = qd_ref[:, 0]
q_final = q_ref[:, N - 1]
qd_final = qd_ref[: N - 1]


STANCE_PHASE_0 = 0
STANCE_PHASE_1 = 0.3 * N
TAKE_OFF_PHASE_0 = 0.3 * N
TAKE_OFF_PHASE_1 = 0.45 * N
FLIGHT_PHASE_0 = 0.45 * N
FLIGHT_PHASE_1 = 0.85 * N
LANDING_PHASE_0 = 0.85 * N
LANDING_PHASE_1 = N

q_i = Q[:, 1]
v_i = V[:, 1]
u_i = U[:, 1]
f_i = F[:, 1]

# qdd_i = fd(q_ref[:,3], qd_ref[:,3], None, U[:,1], F[:,1])
# printSize(qdd_i)


for i in range(N):

    q_i = Q[:, i]
    qd_i = V[:, i]
    u_i = U[:, i]
    f_i = F[:, i]

    q_i_plus_1 = Q[:,i+1]
    qd_i_plus_1 = V[:,i+1]
    qdd

for i in range(N):
    
    # dynamics constraints
    opti.subject_to() 
 
    if STANCE_PHASE_0 <= i < STANCE_PHASE_1:
        # stand phase to set its orientation before take off
        opti.subject_to()

    if TAKE_OFF_PHASE_0 <= i < TAKE_OFF_PHASE_1:
        # take off
        opti.subject_to()

    if FLIGHT_PHASE_0 <= i < FLIGHT_PHASE_1:
        # flight mode
        opti.subject_to()

    if LANDING_PHASE_0 <= i < LANDING_PHASE_1:
        # is in landing phase
        # TODO: how the robot detect if it's landing? How does it know if any of its legs are in contact?
        # TODO: apply whole-body control when landing for stability
        opti.subject_to()

for i in range(N):
    cost_func += 1

opti.minimize(cost_func)


opti.solve()

