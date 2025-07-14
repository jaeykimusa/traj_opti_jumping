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
import time
q_ref = np.array([
    -0.0,  0,  0.33, 0,0,0,
    0,0.806, -1.802,
    0,0.806, -1.802,
    0,0.806, -1.802,
    0,0.806, -1.802,
])
qd_ref = np.zeros_like(q_ref)
f_ref = np.zeros(12)

opt = ca.Opti()

N  = 50
TIMESTEP = 0.05 # seconds
TOTAL_TIME = N * TIMESTEP # 2 seconds

# decision variables
q_opt = opt.variable(NUM_Q, N) 
qd_opt = opt.variable(NUM_Q, N)
qdd_opt = opt.variable(NUM_Q, N)
u_opt = opt.variable(NUM_Q, N)
f_opt = opt.variable(NUM_F, N)

costFunction = 0
ZERO_WEIGHT = 0.0
Q_BASE_POSITION_WEIGHT = 0.01
Q_BASE_ORIENTATION_WEIGHT = 10
Q_JOINT_WEIGHT = 10
QD_BASE_POSITION_WEIGHT = 10
QD_BASE_ORIENTATION_WEIGHT = 5
QD_JOINT_WEIGHT = 1
QDD_BASE_POSITION_WEIGHT = 1
QDD_BASE_ORIENTATION_WEIGHT = 0.5
QDD_JOINT_WEIGHT = 0.1
U_TAU_WEIGHT = 0.01

Q_COST_WEIGHT = np.diag([Q_BASE_POSITION_WEIGHT]*3 + [Q_BASE_ORIENTATION_WEIGHT]*3 + [Q_JOINT_WEIGHT]*12)
QD_COST_WEIGHT = np.diag([QD_BASE_POSITION_WEIGHT]*3 + [QD_BASE_ORIENTATION_WEIGHT]*3 + [QD_JOINT_WEIGHT]*12)
QDD_COST_WEIGHT = np.diag([QDD_BASE_POSITION_WEIGHT]*3 + [QDD_BASE_ORIENTATION_WEIGHT]*3 + [QDD_JOINT_WEIGHT]*12)
U_COST_WEIGHT = np.diag([ZERO_WEIGHT]*6 + [U_TAU_WEIGHT]*12)
F_COST_WEIGHT = np.diag([0.001]*12)


debug = False
if debug:
    STANCE_PHASE_0 = 0
    STANCE_PHASE_1 = N
    FLIGHT_PHASE_0 = N
    FLIGHT_PHASE_1 = N
    LANDING_PHASE_0 = N
    LANDING_PHASE_1 = N
else:
    STANCE_PHASE_0 = 0
    STANCE_PHASE_1 = 0.3 * N
    FLIGHT_PHASE_0 = 0.3 * N
    FLIGHT_PHASE_1 = 0.6 * N
    LANDING_PHASE_0 = 0.6 * N
    LANDING_PHASE_1 = N

# constraints
FZ_MAX = INFINITY
TAU_MAX = 45 # Nm
JOINT_LOWER = np.array([-0.7, -1.5, -1.5] * 4)
JOINT_UPPER = np.array([0.7, 1.5, 1.5] * 4)

# initial guess
q_initial = q_ref
qd_initial = qd_ref
f_initial = f_ref
u_initial = np.zeros(18)
qdd_initial = np.zeros(18)
# u_initial = id(q_initial, qd_initial, qdd_initial, f_initial) # TODO: need more accurate initial guess for u and qdd
# qdd_initial = fd(q_initial, qd_initial, u_initial, f_initial)

def split_fk(fkvec):
    fkvec = fkvec.reshape((-1, 1))
    com_position = fkvec[0:3]
    lf_position = fkvec[3:6]
    rf_position = fkvec[6:9]
    lh_position = fkvec[9:12]
    rh_position = fkvec[12:15]
    return com_position, lf_position, rf_position, lh_position, rh_position

for k in range(N):
    opt.subject_to(qdd_opt[:,k] == fd(q_opt[:,k], qd_opt[:,k], u_opt[:,k], f_opt[:,k]))
    opt.subject_to(u_opt[:6, k] == 0)

    if k == 0:
        opt.subject_to(q_opt[:,k] == q_ref)

    if k == N-1:
        opt.subject_to(q_opt[:,k] == q_ref)

    if k < N-1:
        opt.subject_to(q_opt[:,k+1] == q_opt[:,k] + qd_opt[:,k]*TIMESTEP)
        opt.subject_to(qd_opt[:,k+1] == qd_opt[:,k] + qdd_opt[:,k]*TIMESTEP)

    # stance phase
    if STANCE_PHASE_0 <= k < STANCE_PHASE_1:
        fx1, fy1, fz1 = f_opt[0,k], f_opt[1,k], f_opt[2,k]
        fx2, fy2, fz2 = f_opt[3,k], f_opt[4,k], f_opt[5,k]
        fx3, fy3, fz3 = f_opt[6,k], f_opt[7,k], f_opt[8,k]
        fx4, fy4, fz4 = f_opt[9,k], f_opt[10,k], f_opt[11,k]
        opt.subject_to(fz1 >= 0)
        opt.subject_to(fz1 <= FZ_MAX)
        opt.subject_to(fx1 <= MU * fz1)
        opt.subject_to(fx1 >= -MU * fz1)
        opt.subject_to(fy1 <= MU * fz1)
        opt.subject_to(fy1 >= -MU * fz1)
        opt.subject_to(fz2 >= 0)
        opt.subject_to(fz2 <= FZ_MAX)
        opt.subject_to(fx2 <= MU * fz2)
        opt.subject_to(fx2 >= -MU * fz2)
        opt.subject_to(fy2 <= MU * fz2)
        opt.subject_to(fy2 >= -MU * fz2)
        opt.subject_to(fz3 >= 0)
        opt.subject_to(fz3 <= FZ_MAX)
        opt.subject_to(fx3 <= MU * fz3)
        opt.subject_to(fx3 >= -MU * fz3)
        opt.subject_to(fy3 <= MU * fz3)
        opt.subject_to(fy3 >= -MU * fz3)
        opt.subject_to(fz4 >= 0)
        opt.subject_to(fz4 <= FZ_MAX)
        opt.subject_to(fx4 <= MU * fz4)
        opt.subject_to(fx4 >= -MU * fz4)
        opt.subject_to(fy4 <= MU * fz4)
        opt.subject_to(fy4 >= -MU * fz4)
        # kinematics
        com_position, lf_position, rf_position, lh_position, rh_position = split_fk(fk(q_initial.copy()))
        print("Desired lf position: ", lf_position.reshape(-1))
        print("Desired rf position: ", rf_position.reshape(-1))
        print("Desired lh position: ", lh_position.reshape(-1))
        print("Desired rh position: ", rh_position.reshape(-1))
        print()
        com_position_opt, lf_position_opt, rf_position_opt, lh_position_opt, rh_position_opt = split_fk(fk(q_opt[:,k]))
        if k == 0:
            pass
            opt.subject_to(com_position == com_position_opt)

        
        opt.subject_to(lf_position == lf_position_opt)
        opt.subject_to(rf_position == rf_position_opt)
        opt.subject_to(lh_position == lh_position_opt)
        opt.subject_to(rh_position == rh_position_opt)

    if FLIGHT_PHASE_0 <= k < FLIGHT_PHASE_1:
        opt.subject_to(f_opt[:,k] == 0)
        print("HERE")

    if LANDING_PHASE_0 <= k < LANDING_PHASE_1:
        fx1, fy1, fz1 = f_opt[0,k], f_opt[1,k], f_opt[2,k]
        fx2, fy2, fz2 = f_opt[3,k], f_opt[4,k], f_opt[5,k]
        fx3, fy3, fz3 = f_opt[6,k], f_opt[7,k], f_opt[8,k]
        fx4, fy4, fz4 = f_opt[9,k], f_opt[10,k], f_opt[11,k]
        opt.subject_to(fz1 >= 0)
        opt.subject_to(fz1 <= FZ_MAX)
        opt.subject_to(fx1 <= MU * fz1)
        opt.subject_to(fx1 >= -MU * fz1)
        opt.subject_to(fy1 <= MU * fz1)
        opt.subject_to(fy1 >= -MU * fz1)
        opt.subject_to(fz2 >= 0)
        opt.subject_to(fz2 <= FZ_MAX)
        opt.subject_to(fx2 <= MU * fz2)
        opt.subject_to(fx2 >= -MU * fz2)
        opt.subject_to(fy2 <= MU * fz2)
        opt.subject_to(fy2 >= -MU * fz2)
        opt.subject_to(fz3 >= 0)
        opt.subject_to(fz3 <= FZ_MAX)
        opt.subject_to(fx3 <= MU * fz3)
        opt.subject_to(fx3 >= -MU * fz3)
        opt.subject_to(fy3 <= MU * fz3)
        opt.subject_to(fy3 >= -MU * fz3)
        opt.subject_to(fz4 >= 0)
        opt.subject_to(fz4 <= FZ_MAX)
        opt.subject_to(fx4 <= MU * fz4)
        opt.subject_to(fx4 >= -MU * fz4)
        opt.subject_to(fy4 <= MU * fz4)
        opt.subject_to(fy4 >= -MU * fz4)
        # kinematics
        com_position, lf_position, rf_position, lh_position, rh_position = split_fk(fk(q_initial.copy()))
        print("Desired lf position: ", lf_position.reshape(-1))
        print("Desired rf position: ", rf_position.reshape(-1))
        print("Desired lh position: ", lh_position.reshape(-1))
        print("Desired rh position: ", rh_position.reshape(-1))
        print()
        com_position_opt, lf_position_opt, rf_position_opt, lh_position_opt, rh_position_opt = split_fk(fk(q_opt[:,k]))
        opt.subject_to(lf_position == lf_position_opt)
        opt.subject_to(rf_position == rf_position_opt)
        opt.subject_to(lh_position == lh_position_opt)
        opt.subject_to(rh_position == rh_position_opt)

        if(k == N-1):
            opt.subject_to(com_position == com_position_opt)

    # if TAKE_OFF_PHASE_0 <= k < TAKE_OFF_PHASE_1:
    #     fx1, fy1, fz1 = f_opt[0,k], f_opt[1,k], f_opt[2,k]
    #     fx2, fy2, fz2 = f_opt[3,k], f_opt[4,k], f_opt[5,k]
    #     fx3, fy3, fz3 = f_opt[6,k], f_opt[7,k], f_opt[8,k]
    #     fx4, fy4, fz4 = f_opt[9,k], f_opt[10,k], f_opt[11,k]
    #     opt.subject_to(fz1 == 0)
    #     opt.subject_to(fz1 == 0)
    #     opt.subject_to(fx1 == 0)
    #     opt.subject_to(fx1 == 0)
    #     opt.subject_to(fy1 == 0)
    #     opt.subject_to(fy1 == 0)
    #     opt.subject_to(fz2 == 0)
    #     opt.subject_to(fz2 == 0)
    #     opt.subject_to(fx2 == 0)
    #     opt.subject_to(fx2 == 0)
    #     opt.subject_to(fy2 == 0)
    #     opt.subject_to(fy2 == 0)
    #     opt.subject_to(fz3 >= 0)
    #     opt.subject_to(fz3 <= FZ_MAX)
    #     opt.subject_to(fx3 <= MU * fz3)
    #     opt.subject_to(fx3 >= -MU * fz3)
    #     opt.subject_to(fy3 <= MU * fz3)
    #     opt.subject_to(fy3 >= -MU * fz3)
    #     opt.subject_to(fz4 >= 0)
    #     opt.subject_to(fz4 <= FZ_MAX)
    #     opt.subject_to(fx4 <= MU * fz4)
    #     opt.subject_to(fx4 >= -MU * fz4)
    #     opt.subject_to(fy4 <= MU * fz4)
    #     opt.subject_to(fy4 >= -MU * fz4)
    #     # kinematics
    #     x_k = fk(q_d[:,k])[9:]
    #     opt.subject_to(x_k == fk(q_opt[:,k])[9:])

    # if FLIGHT_PHASE_0 <= k < FLIGHT_PHASE_1:
    #     opt.subject_to(f_opt[:,k] == 0)

    # if LANDING_PHASE_0 <= k < LANDING_PHASE_1:
    #     fx1, fy1, fz1 = f_opt[0,k], f_opt[1,k], f_opt[2,k]
    #     fx2, fy2, fz2 = f_opt[3,k], f_opt[4,k], f_opt[5,k]
    #     fx3, fy3, fz3 = f_opt[6,k], f_opt[7,k], f_opt[8,k]
    #     fx4, fy4, fz4 = f_opt[9,k], f_opt[10,k], f_opt[11,k]
    #     opt.subject_to(fz1 >= 0)
    #     opt.subject_to(fz1 <= FZ_MAX)
    #     opt.subject_to(fx1 <= MU * fz1)
    #     opt.subject_to(fx1 >= -MU * fz1)
    #     opt.subject_to(fy1 <= MU * fz1)
    #     opt.subject_to(fy1 >= -MU * fz1)
    #     opt.subject_to(fz2 >= 0)
    #     opt.subject_to(fz2 <= FZ_MAX)
    #     opt.subject_to(fx2 <= MU * fz2)
    #     opt.subject_to(fx2 >= -MU * fz2)
    #     opt.subject_to(fy2 <= MU * fz2)
    #     opt.subject_to(fy2 >= -MU * fz2)
    #     opt.subject_to(fz3 >= 0)
    #     opt.subject_to(fz3 <= FZ_MAX)
    #     opt.subject_to(fx3 <= MU * fz3)
    #     opt.subject_to(fx3 >= -MU * fz3)
    #     opt.subject_to(fy3 <= MU * fz3)
    #     opt.subject_to(fy3 >= -MU * fz3)
    #     opt.subject_to(fz4 >= 0)
    #     opt.subject_to(fz4 <= FZ_MAX)
    #     opt.subject_to(fx4 <= MU * fz4)
    #     opt.subject_to(fx4 >= -MU * fz4)
    #     opt.subject_to(fy4 <= MU * fz4)
    #     opt.subject_to(fy4 >= -MU * fz4)
    #     # kinematics
    #     x_k = fk(q_d[:,k])
    #     opt.subject_to(x_k == fk(q_opt[:,k]))
    
    # actuation and joint limits
    # opt.subject_to(u_opt[6:18, k] <= TAU_MAX)
    # opt.subject_to(u_opt[6:18, k] >= -TAU_MAX)
    # opt.subject_to(q_opt[6:18, k] <= JOINT_UPPER.reshape(-1, 1))
    # opt.subject_to(q_opt[6:18, k] >= JOINT_LOWER.reshape(-1, 1))

    # # cost accumulation
    costFunction += (q_opt[:,k] - q_ref).T @ Q_COST_WEIGHT @ (q_opt[:,k] - q_ref)
    costFunction += (qd_opt[:,k] - qd_ref).T @ QD_COST_WEIGHT @ (qd_opt[:,k] - qd_ref)
    # costFunction += qdd_opt[:,k].T @ QDD_COST_WEIGHT @ qdd_opt[:,k]
    costFunction += u_opt[:,k].T @ U_COST_WEIGHT @ u_opt[:,k]
    costFunction += (f_opt[:,k] - f_ref).T @ F_COST_WEIGHT @ (f_opt[:,k] - f_ref)

# opt.set_initial(q_opt[:,0], q_initial)
# opt.set_initial(qd_opt[:,0], qd_initial)
# opt.set_initial(qdd_opt[:,0], qd_initial)
# opt.set_initial(u_opt[:,0], u_initial)
# opt.set_initial(f_opt[:,0], f_initial)


opt.minimize(costFunction)

opt.solver("ipopt", {"expand": False}, {"max_iter": 10000})

try:
    print("FULL-BODY TRAJECTORY OPTIMIZATION")
    start = time.time()
    print("OPTIMIZING...")
    sol = opt.solve()
    end = time.time()
    print("\n=== OPTIMIZATION SUCCESSFUL ===")
    print(f"Optimization solve time: {end - start:.2f} seconds")
    Q_OPT = sol.value(q_opt)
    QD_OPT = sol.value(qd_opt)
    QDD_OPT = sol.value(qdd_opt)
    U_OPT = sol.value(u_opt)
    F_OPT = sol.value(f_opt)
    COST_OPTIMAL = sol.value(costFunction)
    
    print(f"\nOptimal Cost: {COST_OPTIMAL:.6f}")

except RuntimeError as e:
    print(f"\n=== OPTIMIZATION FAILED ===")
    print(f"Error: {e}")
    exit()

print(Q_OPT[:,0])
print(Q_OPT[:,1])
print(Q_OPT[:,2])
print(F_OPT[:,0])
print(F_OPT[:,1])


# fig, axs = plt.subplots(3, 3, figsize=(15, 8))  # Wider layout
# axs = axs.flatten()

# axs[0].plot(q_ref[0,:], label="x_d")
# axs[0].plot(q_ref[2,:], label="z_d")
# axs[0].plot(Q_OPT[0,:], label="x_opt")
# axs[0].plot(Q_OPT[2,:], label="z_opt")
# axs[0].set_title("desired vs optimal position")
# axs[0].legend()
# axs[0].set_xlabel("N")
# axs[0].set_ylabel("m")

# axs[1].plot(q_ref[4,:], label="ry_d")
# axs[1].plot(Q_OPT[4,:], label="ry_opt")
# axs[1].set_title("desired vs optimal orientation")
# axs[1].legend()
# axs[1].set_xlabel("N")
# axs[1].set_ylabel("rad")

# axs[2].plot(f_ref[0,:], label="fx_front_d")
# axs[2].plot(f_ref[2,:], label="fz_front_d")
# axs[2].plot(f_ref[6,:], label="fx_rear_d")
# axs[2].plot(f_ref[8,:], label="fz_rear_d")
# axs[2].plot(F_OPT[0,:], label="fx_front_opt")
# axs[2].plot(F_OPT[2,:], label="fz_front_opt")
# axs[2].plot(F_OPT[6,:], label="fx_rear_opt")
# axs[2].plot(F_OPT[8,:], label="fz_rear_opt")
# axs[2].legend()
# axs[2].set_title("desired vs optimal forces")
# axs[2].set_xlabel("N")
# axs[2].set_ylabel("N")

# axs[3].plot(qd_ref[0,:], label="vx_d")
# axs[3].plot(qd_ref[2,:], label="vy_d")
# axs[3].plot(QD_OPT[0,:], label="vx_opt")
# axs[3].plot(QD_OPT[2,:], label="vy_opt")
# axs[3].legend()
# axs[3].set_title("desired vs optimal spatial velocity")
# axs[3].set_xlabel("N")
# axs[3].set_ylabel("m/s")

# axs[4].plot(qd_ref[4,:], label="qd_orientation_d")
# axs[4].plot(QD_OPT[4,:], label="qd_orientation_opt")
# axs[4].legend()
# axs[4].set_title("desired vs optimal angular velocity")
# axs[4].set_xlabel("N")
# axs[4].set_ylabel("rad/s")

# axs[5].plot(qd_ref[7,:], label="qd_left shoulder_d")
# axs[5].plot(qd_ref[8,:], label="qd_left knee_d")
# axs[5].plot(qd_ref[10,:], label="qd_right shoulder_d")
# axs[5].plot(qd_ref[11,:], label="qd_right knee_d")
# axs[5].plot(QD_OPT[7,:], label="qd_left shoulder_opt")
# axs[5].plot(QD_OPT[8,:], label="qd_left knee_opt")
# axs[5].plot(QD_OPT[10,:], label="qd_right shoulder_opt")
# axs[5].plot(QD_OPT[11,:], label="qd_right knee_opt")
# axs[5].legend()
# axs[5].set_title("desired vs optimal joint velocity for front leg")
# axs[5].set_xlabel("N")
# axs[5].set_ylabel("rad/s")

# axs[6].plot(qd_ref[13,:], label="qd_left shoulder_d")
# axs[6].plot(qd_ref[14,:], label="qd_left knee_d")
# axs[6].plot(qd_ref[16,:], label="qd_right shoulder_d")
# axs[6].plot(qd_ref[17,:], label="qd_right knee_d")
# axs[6].plot(QD_OPT[13,:], label="qd_left shoulder_opt")
# axs[6].plot(QD_OPT[14,:], label="qd_left knee_opt")
# axs[6].plot(QD_OPT[16,:], label="qd_right shoulder_opt")
# axs[6].plot(QD_OPT[17,:], label="qd_right knee_opt")
# axs[6].legend()
# axs[6].set_title("desired vs optimal joint velocity for rear leg")
# axs[6].set_xlabel("N")
# axs[6].set_ylabel("rad/s")

# plt.tight_layout()
# plt.show()
# exit()








# vis via rerun
import rerun as rr
from go2.mpac_logging.mpac_logging import robot_zoo
from go2.mpac_logging.mpac_logging.rerun.robot_logger import RobotLogger
from go2.mpac_logging.mpac_logging.rerun.utils import rerun_initialize, rerun_store

rr.init("simple_robot_example", spawn=False)
# robot_logger = RobotLogger.from_zoo("go2")
robot_logger = RobotLogger.from_zoo("go2_description")

import time

rerun_initialize("forward jump test", spawn=True)
print("works")

current_time = time.time()
# robot_logger.log_initial_state(logtime=current_time)

dt = TIMESTEP

for i in range(N):
    q_i = Q_OPT[:,i]
    print("Forward kinematics: ", fk(q_i)[-15:])
    base_position = q_i[:3]
    base_orientation = R.from_euler("xyz", q_i[3:6], degrees=False).as_quat()
    joint_positions = {
        "FL_hip_joint": q_i[6], 
        "FL_thigh_joint": q_i[7],
        "FL_calf_joint": q_i[8],
        "RR_hip_joint": q_i[9],
        "RR_thigh_joint": q_i[10],
        "RR_calf_joint": q_i[11],
        "FR_hip_joint": q_i[12],
        "FR_thigh_joint": q_i[13],
        "FR_calf_joint": q_i[14],
        "RL_hip_joint": q_i[15],
        "RL_thigh_joint": q_i[16],
        "RL_calf_joint": q_i[17],
    }

    print(q_i)

    robot_logger.log_state(
        logtime=current_time,
        base_position=base_position,
        base_orientation=base_orientation,
        joint_positions=joint_positions
    )
    current_time += dt

# rr.save("forward_jump_test_today.rrd")

exit()