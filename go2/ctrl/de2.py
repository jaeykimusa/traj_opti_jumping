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

q0 = getDefaultStandState(model, data)
v0 = np.zeros(18)
a0 = np.zeros(18)
g_grav = pin.rnea(model, data, q0, v0, a0)
print(g_grav)

exit()

tau = pin.rnea(model, data, q, v, qdd)
f = list(computeFullContactForces(model, data, q, v, qdd))

cs_q = ca.SX.sym("q", NUM_Q, 1)
cs_v = ca.SX.sym("qd", NUM_Q, 1)
cs_qdd = ca.SX.sym("qdd", NUM_Q, 1)
cs_tau = ca.SX.sym("u", NUM_Q, 1)
cs_f = ca.SX.sym("f", NUM_F, 1)

pinocchio.casadi.aba(ad_model, ad_data, cs_q, cs_v, cs_tau)
a_ad = ad_data.ddq

eval_aba = ca.Function("eval_aba", [cs_q, cs_v, cs_tau, cs_f], [a_ad])

# # Evaluate CasADi expression with real value
# a_casadi_res = eval_aba(q, v, tau)

# # Eval ABA using classic Pinocchio model
pin.aba(model, data, q, v, tau, f)

print(data.ddq.T)
exit()

g_bl = g_grav[:6]
g_j = g_grav[6:]

# finding contacts
feet_ids = [Frame.FL_EE, Frame.FR_EE, Frame.RL_EE, Frame.RR_EE]
bl_id = model.getFrameId("")