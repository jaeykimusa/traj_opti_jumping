from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.dynamics.id import *
from go2.utils.math_utils import *
# from go2.utils.io_utils import *
import casadi as ca
import matplotlib.pyplot as plt

import os
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R

from pathlib import Path
from sys import argv
from enum import Enum, auto

from go2.robot.morphology import *
import numpy as np


q = getDefaultStandState(model, data)
v = np.zeros(18)
qdd = np.zeros(18)
tau = pin.rnea(model, data, q, v, qdd)
f = computeFullContactForces(model, data, q, v, qdd)
u = tau + computeContactJacobian(model, data, q).T @ f

cs_q = ca.SX.sym("q", NUM_Q, 1)
cs_v = ca.SX.sym("qd", NUM_Q, 1)
cs_qdd = ca.SX.sym("qdd", NUM_Q, 1)
cs_u = ca.SX.sym("u", NUM_Q, 1)
cs_tau = ca.SX.sym("u", NUM_Q, 1)
cs_f = ca.SX.sym("f", NUM_F, 1)
cs_J_c = ca.SX.sym("J_c", NUM_F, NUM_Q)

opti = ca.Opti()

q_opt = opti.variable(NUM_Q, 1) 
v_opt = opti.variable(NUM_Q, 1) 
tau_opt = opti.variable(NUM_Q, 1) 
f_opt = opti.variable(NUM_F, 1)

q_desired = opti.parameter(NUM_Q, 1)
v_desired = opti.parameter(NUM_Q, 1)
tau_desired = opti.parameter(NUM_Q, 1)
f_desired = opti.parameter(NUM_F, 1)

ddq_desired = opti.parameter(NUM_Q, 1)

ddq_sym = id(q_opt, v_opt, tau_opt, f_opt)

cost = ca.sumsqr(ddq_sym - ddq_desired) * 10.0 \
       + ca.sumsqr(q_opt - q_desired) * 42 \
       + ca.sumsqr(v_opt - v_desired) * 42 \
       + ca.sumsqr(tau_opt - tau_desired) * 42 \
       + ca.sumsqr(f_opt - f_desired) * 42

opti.minimize(cost)

opti.subject_to(v_opt >= -np.ones((NUM_Q,1)) * 50) # Example: +/- 50 rad/s or m/s
opti.subject_to(v_opt <= np.ones((NUM_Q,1)) * 50)
opti.subject_to(tau_opt >= -np.ones((NUM_Q,1)) * 10000) # Example: +/- 100 Nm
opti.subject_to(tau_opt <= np.ones((NUM_Q,1)) * 10000)
opti.subject_to(f_opt >= -np.ones((NUM_F,1)) * 10000) # Example: +/- 100 Nm
opti.subject_to(f_opt <= np.ones((NUM_F,1)) * 10000)

ddq_desired_numerical = data.ddq
q_nominal_numerical = q
v_nominal_numerical = v
tau_nominal_numerical = tau
f_nominal_numerical = f
u_nominal_numerical = u

opti.set_value(ddq_desired, ddq_desired_numerical)
opti.set_value(q_desired, q_nominal_numerical)
opti.set_value(v_desired, v_nominal_numerical)
opti.set_value(tau_desired, tau_nominal_numerical)
opti.set_value(f_desired, f_nominal_numerical)

opti.set_initial(q_opt, np.zeros((NUM_Q, 1)))
opti.set_initial(v_opt, np.zeros((NUM_Q, 1)))
opti.set_initial(tau_opt, np.zeros((NUM_Q, 1)))
opti.set_initial(f_opt, np.zeros((NUM_F, 1)))


opti.solver("ipopt", {"expand":True}, {"max_iter":1000})
# sol = opti.solve()

try:
    sol = opti.solve()
except RuntimeError as e:
    print(f"IPOPT failed to converge: {e}")
    print("q_opt debug:", opti.debug.value(q_opt))
    print(q)
    print("v_opt debug:", opti.debug.value(v_opt))
    print(v)
    print("tau_opt debug:", opti.debug.value(tau_opt))
    print(tau)
    print("f_opt debug:", opti.debug.value(f_opt))
    print(f)
    exit()

q_optimal = sol.value(q_opt)
v_optimal = sol.value(v_opt)
tau_optimal = sol.value(tau_opt)
f_optimal = sol.value(f_opt)
cost_optimal = sol.value(cost)

# numerical_data = model.createData()
pin.aba(model, data, q_nominal_numerical, v_nominal_numerical, u_nominal_numerical)
ddq_resulting_optimal = data.ddq

print("\n--- Optimization Results (Single Time Instance) ---")
print(f"Optimal Cost: {cost_optimal}")
print(f"Desired Acceleration (ddq_des):\n{ddq_desired_numerical.T}")
print(f"Optimal Configuration (q_opt):\n{q_optimal.T}")
print(f"Optimal Velocity (v_opt):\n{v_optimal.T}")
print(f"Optimal Torques (u_opt):\n{tau_optimal.T}")
print(f"Optimal Torques (u_opt):\n{f_optimal.T}")
print(f"Resulting Acceleration (ddq_actual):\n{ddq_resulting_optimal.T}")


# Verify how close we got to the desired acceleration
print(f"\nError in achieved ddq vs. desired ddq (norm): {np.linalg.norm(ddq_resulting_optimal - ddq_desired_numerical)}")


fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # Wider layout

# Flatten the 2D array of axes for easy indexing
axs = axs.flatten()

# First subplot: qdd comparison
axs[0].plot(ddq_resulting_optimal, label="qdd_optimized")
axs[0].plot(ddq_desired_numerical, label="qdd_desired")
axs[0].set_title("qdd comparison")
axs[0].legend()

# Second subplot: q comparison
axs[1].plot(q_optimal, label="q_optimized")
axs[1].plot(q_nominal_numerical, label="q_desired")
axs[1].set_title("q comparison")
axs[1].legend()

# Third subplot: v comparison
axs[2].plot(v_optimal, label="v_optimal")
axs[2].plot(v_nominal_numerical, label="v_desired")
axs[2].set_title("v comparison")
axs[2].legend()

# Fourth subplot: tau comparison
axs[3].plot(tau_optimal, label="tau_optimal")
axs[3].plot(tau_nominal_numerical, label="tau_desired")
axs[3].set_title("tau comparison")
axs[3].legend()

# Fifth subplot: f comparison
axs[4].plot(f_optimal, label="f_optimal")
axs[4].plot(f_nominal_numerical, label="f_desired")
axs[4].set_title("f comparison")
axs[4].legend()

Jc_F_optimal= computeFullContactJacobians(q_optimal).T @ f_nominal_numerical
Jc_F_nominal_numerical = computeFullContactJacobians(q_nominal_numerical).T @ f_nominal_numerical
# 6 subplot: Jc comparison
axs[5].plot(Jc_F_optimal, label="Jc_F_optimal")
axs[5].plot(Jc_F_nominal_numerical, label="Jc_F_desired")
axs[5].set_title("Jc*F comparison")
axs[5].legend()

# Optional: label axes, add spacing
for i in range(5):
    axs[i].set_xlabel("Time step")
    axs[i].set_ylabel("Value")

plt.tight_layout()
plt.show()







