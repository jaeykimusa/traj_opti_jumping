from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from go2.kinematics.kinematics import *
from go2.dynamics.dynamics import *
from go2.dynamics.fd import *
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

# This path refers to pin source code but you can define your own directory here.
pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
 
# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = (
    pin_model_dir / "model/go2/go2.urdf"
    if len(argv) < 2
    else argv[1]
)
 
# # Load the urdf model

joint_model = pin.JointModelComposite(2)
joint_model.addJoint(pin.JointModelTranslation())
joint_model.addJoint(pin.JointModelSphericalZYX())

# model = pin.buildModelFromUrdf(urdf_filename, joint_model)
# robot = pin.RobotWrapper(model)


# symbolic term
import pinocchio.casadi
model = pin.buildModelFromUrdf(urdf_filename, joint_model)
robot = pin.RobotWrapper(model)
data = model.createData()

q = np.linspace(1, 10, 18)
v = np.linspace(1, 10, 18)
tau = np.linspace(1, 10, 18)

cs_q = ca.SX.sym("q", NUM_Q, 1)
cs_v = ca.SX.sym("qd", NUM_Q, 1)
qdd = ca.SX.sym("qdd", NUM_Q, 1)
cs_tau = ca.SX.sym("u", NUM_Q, 1)

ad_model = pinocchio.casadi.Model(model)
ad_data = ad_model.createData()

pinocchio.casadi.aba(ad_model, ad_data, cs_q, cs_v, cs_tau)
a_ad = ad_data.ddq

eval_aba = ca.Function("eval_aba", [cs_q, cs_v, cs_tau], [a_ad])

# # Evaluate CasADi expression with real value
# a_casadi_res = eval_aba(q, v, tau)

# # Eval ABA using classic Pinocchio model
# pin.aba(model, data, q, v, tau)

# Print both results
# print("pinocchio double:\n\ta =", data.ddq.T)
# print("pinocchio CasADi:\n\ta =", np.array(a_casadi_res).T)

# Verify results are close
# np.testing.assert_allclose(data.ddq, a_casadi_res, atol=1e-9)
# print("\nResults from Pinocchio (double) and Pinocchio (CasADi) match.")

# qdd123 = pin.aba(model_casadi, data_casadi, q, qd, u)

opti = ca.Opti()

q_opt = opti.variable(NUM_Q, 1) 
v_opt = opti.variable(NUM_Q, 1) 
u_opt = opti.variable(NUM_Q, 1) 

q_desired = opti.parameter(NUM_Q, 1)
v_desired = opti.parameter(NUM_Q, 1)
u_desired = opti.parameter(NUM_Q, 1)

ddq_desired = opti.parameter(NUM_Q, 1)

ddq_sym = eval_aba(q_opt, v_opt, u_opt)

cost = ca.sumsqr(ddq_sym - ddq_desired) * 100.0 \
       + ca.sumsqr(q_opt - q_desired) * 1e-3 \
       + ca.sumsqr(v_opt - v_desired) * 1e-3 \
       + ca.sumsqr(u_opt - u_desired) * 1e-4

opti.minimize(cost)

opti.subject_to(q_opt >= model.lowerPositionLimit)
opti.subject_to(q_opt <= model.upperPositionLimit)
opti.subject_to(v_opt >= -np.ones((NUM_Q,1)) * 50) # Example: +/- 50 rad/s or m/s
opti.subject_to(v_opt <= np.ones((NUM_Q,1)) * 50)
opti.subject_to(u_opt >= -np.ones((NUM_Q,1)) * 100) # Example: +/- 100 Nm
opti.subject_to(u_opt <= np.ones((NUM_Q,1)) * 100)

ddq_desired_numerical = data.ddq
q_nominal_numerical = q
v_nominal_numerical = v
u_nominal_numerical = tau

opti.set_value(ddq_desired, ddq_desired_numerical)
opti.set_value(q_desired, q_nominal_numerical)
opti.set_value(v_desired, v_nominal_numerical)
opti.set_value(u_desired, u_nominal_numerical)

opti.set_initial(q_opt, np.zeros((NUM_Q, 1)))
opti.set_initial(v_opt, np.zeros((NUM_Q, 1)))
opti.set_initial(u_opt, np.zeros((NUM_Q, 1)))

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
    print("u_opt debug:", opti.debug.value(u_opt))
    exit()

q_optimal = sol.value(q_opt)
v_optimal = sol.value(v_opt)
u_optimal = sol.value(u_opt)
cost_optimal = sol.value(cost)

numerical_data = model.createData()
pin.aba(model, numerical_data, q_optimal, v_optimal, u_optimal)
ddq_resulting_optimal = numerical_data.ddq

print("\n--- Optimization Results (Single Time Instance) ---")
print(f"Optimal Cost: {cost_optimal}")
print(f"Desired Acceleration (ddq_des):\n{ddq_desired_numerical.T}")
print(f"Optimal Configuration (q_opt):\n{q_optimal.T}")
print(f"Optimal Velocity (v_opt):\n{v_optimal.T}")
print(f"Optimal Torques (u_opt):\n{u_optimal.T}")
print(f"Resulting Acceleration (ddq_actual):\n{ddq_resulting_optimal.T}")

# Verify how close we got to the desired acceleration
print(f"\nError in achieved ddq vs. desired ddq (norm): {np.linalg.norm(ddq_resulting_optimal - ddq_desired_numerical)}")