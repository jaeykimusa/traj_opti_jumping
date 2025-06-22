# stand_idqp.py

# go2_stand.py

from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.robot.robot import * 
from go2.utils.math_utils import *

# =================================================================
#   PRIMARY GOAL IS TO SOLVE QP USING INVERSE DYNAMICS TO STAND
# =================================================================

import scipy.sparse as sp
import osqp

def printSize(matrix):
    print(matrix.shape)

# init
q = getDefaultStandState(model, data)
pin.framesForwardKinematics(model, data, q)

qd = np.zeros(model.nv)
qdd = np.zeros(model.nv)

# Fc_FL, Fc_FR, Fc_RL, Fc_RR = computeContactForces(model, data, q, qd, qdd)
# printContactForces(Fc_FL, Fc_FR, Fc_RL, Fc_RR)

idqp = osqp.OSQP()

num_var = NUM_U + NUM_Q + NUM_C * 3

mu = 0.8

# decision variables, let x = [u, qdd, f].T
x = getZerosMatrix(num_var, 1)
# Hessian matrix, error weight for the quadratic cost
Q_cost = getZerosMatrix(num_var, num_var)
for i in range(0, NUM_U):
    Q_cost[i, i] = 1
for i in range(NUM_U, NUM_U + 6):
    Q_cost[i, i] = 1
for i in range(NUM_U + 6, NUM_U + NUM_Q):
    Q_cost[i, i] = 0.0001
for i in range(NUM_U + NUM_Q, num_var):
    Q_cost[i, i] = 0.001

# gradient vector, error weight for the linear part of the cost
c_cost = getZerosMatrix(num_var, 1)

# equality constraints: Ax = b
A_eq = getZerosMatrix(30, num_var)
A_eq_upper = getZerosMatrix(18, 42)
A_eq_lower = getZerosMatrix(12, 42)
A_eq_zeros_block = getZerosMatrix(12, 12)
M_term = getMassInertiaMatrix(q)
# printSize(M_term)
CG_terms = getCoriolisGravity(q, qd, qdd)
# printSize(CG_terms)
# The distribution matrix of actuator torques
B_term = np.vstack([getZerosMatrix(6, 12), getIdentityMatrix(12)])
Jc = computeContactJacobian(model, data, q)
A_eq_upper = np.hstack([B_term, M_term, Jc.T])
A_eq_lower = np.hstack([A_eq_zeros_block, Jc, A_eq_zeros_block])
A_eq = np.vstack([A_eq_upper, A_eq_lower])
printSize(A_eq)
b_eq = getZerosMatrix(30, 1)
b_eq_upper = getZerosMatrix(18, 1)
b_eq_lower = getZerosMatrix(12, 1)
b_eq_upper = CG_terms
Jdc = computeContactJacobiansTimeVariation(q, qd)
b_eq_lower = -Jdc * qd
b_eq = np.vstack([b_eq_upper, b_eq_lower])



# inequality constraints: Ax <= b
A_ineq = getZerosMatrix(20, 42)
A_ineq_zeros = getZerosMatrix(20, 30)
A_ineq_zeros_block = getZerosMatrix(5, 3)
A_ineq_c = getZerosMatrix(5, 3)
A_ineq_c[0,0] = 1
A_ineq_c[0,2] = -mu
A_ineq_c[1,0] = -1
A_ineq_c[1,2] = -mu
A_ineq_c[2,1] = 1
A_ineq_c[2,2] = -mu
A_ineq_c[3,1] = -1
A_ineq_c[3,2] = -mu
A_ineq_c[4,2] = -1
A_ineq_FL_block = np.vstack([A_ineq_c, A_ineq_zeros_block, A_ineq_zeros_block, A_ineq_zeros_block])
A_ineq_FR_block = np.vstack([A_ineq_zeros_block, A_ineq_c, A_ineq_zeros_block, A_ineq_zeros_block])
A_ineq_RL_block = np.vstack([A_ineq_zeros_block, A_ineq_zeros_block, A_ineq_c, A_ineq_zeros_block])
A_ineq_RR_block = np.vstack([A_ineq_zeros_block, A_ineq_zeros_block, A_ineq_zeros_block, A_ineq_c])
A_ineq = np.hstack([A_ineq_zeros, A_ineq_FL_block, A_ineq_FR_block, A_ineq_RL_block, A_ineq_RR_block])
b_ineq = np.zeros((20,1))
# printSize(A_ineq)
# printSize(b_ineq)

# boundaries
x_min = np.vstack([np.full((12, 1), -15), getZerosMatrix(30, 1)])
x_max = np.vstack([np.full((12, 1), 15), getZerosMatrix(30, 1)])

A_full = np.vstack([A_eq, A_ineq])
x_min_full = np.hstack([b_eq, -100000000000]) # infinity lower bounds for inequality constraints
x_max_full = np.hstack([b_eq, b_ineq])


idqp.setup(P=Q_cost, q=c_cost.T, A=A_full, l=x_min_full, u=x_max_full, verbose=False)
idqp_sol = idqp.solve()
sol_x = idqp_sol.x
u = sol_x[:NUM_U]
qdd = sol_x[NUM_U:NUM_U+NUM_Q]
F = sol_x[NUM_U+NUM_Q:] #.reshape(NUM_C, 3)


# Jcd = computeContactJacobiansTimeVariation(model, data, q, qd)
# printSize(Jcd)
# print(Jcd)



# class StandIDQP:
#     def __init__(self, robot_model):

    

def standIdqp(*arg):
    x_d = arg[0]
    y_d = arg[1]
    z_d = arg[2]
    rx_d = arg[3]
    ry_d = arg[4]
    rz_d = arg[5]







