# jump_with_jakub.py

from pathlib import Path
from sys import argv

from go2.kinematics.kinematics import *
from go2.kinematics.fk import *
from go2.dynamics.dynamics import *
from go2.dynamics.fd import *
from go2.dynamics.id import *
from go2.utils.math_utils import *
from go2.vis.rerun import *
from go2.robot.morphology import *
from go2.utils.io_utils import *

import matplotlib.pyplot as plt

q_ref, qd_ref, f_ref = readReferenceData("q_ref_forward_jump.txt", "v_ref.txt", "f_ref.txt", "./go2/ctrl/data/")

N = 100
TIMESTEP = 0.02 # seconds

STANCE_PHASE_0 = 0
STANCE_PHASE_1 = 0.3 * N
TAKE_OFF_PHASE_0 = 0.3 * N
TAKE_OFF_PHASE_1 = 0.45 * N
FLIGHT_PHASE_0 = 0.45 * N
FLIGHT_PHASE_1 = 0.85 * N
LANDING_PHASE_0 = 0.85 * N
LANDING_PHASE_1 = N








# q = getDefaultStandState(model, data)
# jump_path = os.path.join(os.path.dirname(__file__), "q_ref_forward_jump.txt")
# q_ref = np.loadtxt(jump_path, delimiter=',')
# if (getRowSize(q_ref) == 7):
#     q_ref_new = getZerosMatrix(NUM_Q, getColumnSize(q_ref))
#     q_ref_new[0, :] = q_ref[0, :]
#     q_ref_new[2, :] = q_ref[1, :]
#     q_ref_new[4, :] = q_ref[2, :]
#     q_ref_new[7, :] = -q_ref[3, :] # FL2
#     q_ref_new[8, :] = -q_ref[4, :] # FL3
#     q_ref_new[10, :] = -q_ref[3, :] # FR2
#     q_ref_new[11, :] = -q_ref[4, :] # FR3
#     q_ref_new[13, :] = -q_ref[5, :] # RL2
#     q_ref_new[14, :] = -q_ref[6, :] # RL3
#     q_ref_new[16, :] = -q_ref[5, :] # RR2
#     q_ref_new[17, :] = -q_ref[6, :] # RR3
#     q_ref = q_ref_new

# plt.plot(q, label="from ik")
# plt.plot(q_ref[:,0], label="from ref")
# plt.title(f"q0 comparison")
# plt.xlabel("")
# plt.ylabel("")
# plt.legend()
# plt.grid(True)
# plt.show()