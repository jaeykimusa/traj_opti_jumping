# jump_full.py

from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.robot.robot import * 
from go2.utils.math_utils import *
import casadi as ca

# =================================================================
#   PRIMARY GOAL IS TO OPTIMIZE FULL BODY TRAJECTORY
# =================================================================

import os
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R

def printSize(matrix):
    print(matrix.shape)

jump_path = os.path.join(os.path.dirname(__file__), "q_ref_forward_jump.txt")
q_ref = np.loadtxt(jump_path, delimiter=',')

# q_ref = interpolateMatrixToTargetColumns(q_ref, 1000)

printSize(q_ref)

dt = 0.02
N = 100

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
q_init = q_ref[:,1]
qd_init = np.zeros((NUM_Q))
q_final = q_ref[:, N - 1]
qd_final = np.zeros((NUM_Q))





