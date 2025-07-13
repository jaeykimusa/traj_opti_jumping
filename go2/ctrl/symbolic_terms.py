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

import matplotlib.pyplot as plt

# Get nominal standing configuration
q = getDefaultStandStateFullOptimization(model, data)
qd = np.zeros(18)
qdd = np.zeros(18)
f = computeFullContactForces(model, data, q, qd, qdd)
u = id(q, qd, qdd, f)

# x = fd(q, qd, u, f)
# print(x)
# exit()
# print("Reference standing configuration:")
# print(f"q_ref: {q}")
# print(f"Orientation (rx,ry,rz): {q[3:6]}")
# print(f"Base height: {q[2]}")
# print(f"Reference joint torques: {tau[6:]}")

opti = ca.Opti()

# decision variables
q_opt = opti.variable(NUM_Q, 1) 
qd_opt = opti.variable(NUM_Q, 1) 
qdd_opt = opti.variable(NUM_Q, 1)
u_opt = opti.variable(NUM_Q, 1) 
f_opt = opti.variable(NUM_F, 1)

# desired parameters
q_d = opti.parameter(NUM_Q, 1)
qd_d = opti.parameter(NUM_Q, 1)
qdd_d = opti.parameter(NUM_Q, 1)
u_d = opti.parameter(NUM_Q, 1)
f_d = opti.parameter(NUM_F, 1)

# s.t.
# opti.subject_to(q_opt == q_d)
opti.subject_to(qd_opt == np.zeros(NUM_Q).reshape(-1, 1))
opti.subject_to(qdd_opt == fd(q_d, qd_d, u_d, f_d))
opti.subject_to(qdd_opt == np.zeros(NUM_Q).reshape(-1, 1))
opti.subject_to(u_opt == id(q_opt, qd_opt, qdd_opt, f_opt))
# opti.subject_to(q_opt[:6] == q_d[:6])

# joint angle constraints
# joint_lower = np.array([-3.14, -1.57, -3.14] * 4)
# joint_upper = np.array([3.14, 1.57, 3.14] * 4)
# opti.subject_to(q_opt[6:18] >= joint_lower.reshape(-1, 1))
# opti.subject_to(q_opt[6:18] <= joint_upper.reshape(-1, 1))

tau_max = 45.0  # Nm
opti.subject_to(u_opt[6:18] >= -tau_max)
opti.subject_to(u_opt[6:18] <= tau_max)

for i in range(4):  # 4 feet
    foot_idx = i * 3
    fx = f_opt[foot_idx]
    fy = f_opt[foot_idx + 1] 
    fz = f_opt[foot_idx + 2]
    
    # Normal force must be positive
    opti.subject_to(fz >= 0.0)   # Minimum normal force
    opti.subject_to(fz <= INFINITY)  # Maximum normal force
    
    # Friction cone constraint
    opti.subject_to(fx <= MU * fz)
    opti.subject_to(fx >= -MU * fz)
    opti.subject_to(fy <= MU * fz)
    opti.subject_to(fy >= -MU * fz)

# # 10. Total vertical force should approximately equal robot weight
# total_fz = ca.sum1(f_opt[2::3])
# robot_weight = getMass(model) * 9.81  # Assuming 12kg robot
# opti.subject_to(total_fz >= robot_weight * 0.95)
# opti.subject_to(total_fz <= robot_weight * 1.05)

# ===== COST FUNCTION =====
cost = 0
COST_WEIGHT_Q = 1
COST_WEIGHT_QD = 0.1
COST_WEIGHT_QDD = 0.1
COST_WEIGHT_U = 0.2
COST_WEIGHT_F = 0.2

# 1. Configuration cost - heavily weight orientation to keep robot upright
# cost += ca.sumsqr(q_opt[3:6]) * 500  # Very high weight on keeping orientation zero
# cost += ca.sumsqr(q_opt[6:] - q_d[6:]) * 100  # Joint angles

# 2. Velocity and acceleration should be exactly zero (add extra penalty)
# cost += ca.sumsqr(qd_opt) * 100
# cost += ca.sumsqr(qdd_opt) * 100

# 3. Joint torque minimization
# cost += ca.sumsqr(u_opt[6:18]) * 10

# # 4. Force distribution
# target_fz = robot_weight / 4
# for i in range(4):
#     cost += (f_opt[i*3 + 2] - target_fz)**2 * 100
#     cost += f_opt[i*3]**2 * 50      # Minimize fx
#     cost += f_opt[i*3 + 1]**2 * 50  # Minimize fy

# 5. Stay close to desired configuration
cost += COST_WEIGHT_Q * (q_opt - q_d).T @ (q_opt - q_d)
cost += COST_WEIGHT_QD * (qd_opt - qd_d).T @ (qd_opt - q_d)
cost += COST_WEIGHT_QDD * (qdd_opt - qdd_d).T @ (qdd_opt - q_d)
cost += COST_WEIGHT_U * (u_opt - u_d).T @ (u_opt - u_d)
cost += COST_WEIGHT_F * (f_opt - f_d).T @ (f_opt - f_d)


opti.minimize(cost)

# Set parameter values
opti.set_value(q_d, q.reshape(-1, 1)) #.reshape(-1, 1))
opti.set_value(qd_d, qd.reshape(-1, 1))
opti.set_value(qdd_d, qdd.reshape(-1, 1))
opti.set_value(u_d, u.reshape(-1, 1))
opti.set_value(f_d, f.reshape(-1, 1))

# Initial guess
opti.set_initial(q_opt, q.reshape(-1, 1))
opti.set_initial(qd_opt, qd.reshape(-1, 1)) #np.zeros((18, 1)))
opti.set_initial(qdd_opt, qdd.reshape(-1, 1)) #np.zeros((18, 1)))
opti.set_initial(u_opt, u.reshape(-1, 1))
opti.set_initial(f_opt, f.reshape(-1, 1))

# Solver options
opti.solver("ipopt", {"expand": True}, {"max_iter": 3000})
#     "max_iter": 3000,
#     "tol": 1e-8,
#     "acceptable_tol": 1e-6,
#     "print_level": 5,
#     "linear_solver": "mumps",
#     "hessian_approximation": "limited-memory",
#     "mu_strategy": "adaptive",
#     "warm_start_init_point": "yes"
# })

try:
    sol = opti.solve()
    print("\n=== OPTIMIZATION SUCCESSFUL ===")
    
    q_optimal = sol.value(q_opt)
    qd_optimal = sol.value(qd_opt)
    qdd_optimal = sol.value(qdd_opt)
    u_optimal = sol.value(u_opt)
    f_optimal = sol.value(f_opt)
    cost_optimal = sol.value(cost)
    
    print(f"\nOptimal Cost: {cost_optimal:.6f}")
    
    print("\n--- Configuration Comparison ---")
    print(f"Base Position - Desired: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}]")
    print(f"Base Position - Optimal: [{q_optimal[0]:.6f}, {q_optimal[1]:.6f}, {q_optimal[2]:.6f}]")
    
    print(f"\nBase Orientation - Desired: [{q[3]:.6f}, {q[4]:.6f}, {q[5]:.6f}]")
    print(f"Base Orientation - Optimal: [{q_optimal[3]:.6f}, {q_optimal[4]:.6f}, {q_optimal[5]:.6f}]")
    
    print("\n--- Dynamics Verification ---")
    print(f"Velocity norm - Desired: {np.linalg.norm(qd):.6f}")
    print(f"Velocity norm - Optimal: {np.linalg.norm(qd_optimal):.6f}")
    
    print(f"\nAcceleration norm - Desired: {np.linalg.norm(qdd):.6f}")
    print(f"Acceleration norm - Optimal: {np.linalg.norm(qdd_optimal):.6f}")
    
    print(f"\nMax joint torque - Desired: {np.max(np.abs(u[6:])):.3f}")
    print(f"Max joint torque - Optimal: {np.max(np.abs(u_optimal[6:])):.3f}")
    
    # Verify dynamics
    u_check = id(q_optimal, qd_optimal, u_optimal, f_optimal)
    print(f"\nDynamics verification error: {np.linalg.norm(u_check - u_optimal):.6e}")
    
    # Force analysis
    print(f"\nContact Forces:")
    total_fz_opt = 0
    for i in range(4):
        fx_opt, fy_opt, fz_opt = f_optimal[i*3:(i+1)*3].flatten()
        fx_des, fy_des, fz_des = f[i*3:(i+1)*3]
        total_fz_opt += fz_opt
        
        print(f"  Foot {i+1}:")
        print(f"    Desired:  fx={fx_des:.2f}, fy={fy_des:.2f}, fz={fz_des:.2f}")
        print(f"    Optimal:  fx={fx_opt:.2f}, fy={fy_opt:.2f}, fz={fz_opt:.2f}")
        
        friction_ratio = np.sqrt(fx_opt**2 + fy_opt**2) / fz_opt
        print(f"    Friction utilization: {friction_ratio/mu*100:.1f}%")
    
    print(f"\nTotal vertical force - Optimal: {total_fz_opt:.2f} N")
    print(f"Expected weight: {12.0 * 9.81:.2f} N")
    
except RuntimeError as e:
    print(f"\n=== OPTIMIZATION FAILED ===")
    print(f"Error: {e}")
    
    # Debug information
    print("\nDebugging Values:")
    q_debug = opti.debug.value(q_opt)
    qd_debug = opti.debug.value(qd_opt)
    qdd_debug = opti.debug.value(qdd_opt)
    u_debug = opti.debug.value(u_opt)
    f_debug = opti.debug.value(f_opt)
    
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 8))  # Wider layout
    axs = axs.flatten()
    # Second subplot: q comparison
    axs[0].plot(q_debug, label="q_debug")
    axs[0].plot(q, label="q_desired")
    axs[0].set_title("q comparison")
    axs[0].legend()
    # Third subplot: v comparison
    axs[1].plot(qd_debug, label="qd_debug")
    axs[1].plot(qd, label="qd_desired")
    axs[1].set_title("qd comparison")
    axs[1].legend()
    # First subplot: qdd comparison
    axs[2].plot(qdd_debug, label="qdd_debug")
    axs[2].plot(qdd, label="qdd_desired")
    axs[2].set_title("qdd comparison")
    axs[2].legend()
    # Fourth subplot: tau comparison
    axs[3].plot(u_debug, label="u_debug")
    axs[3].plot(u, label="u_desired")
    axs[3].set_title("u comparison")
    axs[3].legend()

    # Fifth subplot: f comparison
    axs[4].plot(f_debug, label="f_debug")
    axs[4].plot(f, label="f_desired")
    axs[4].set_title("f comparison")
    axs[4].legend()

    Jc_F_debug= computeFullContactJacobians(q_debug).T @ f_debug
    Jc_F_desired = computeFullContactJacobians(q).T @ f
    # 6 subplot: Jc comparison
    axs[5].plot(Jc_F_debug, label="Jc_F_debug")
    axs[5].plot(Jc_F_desired, label="Jc_F_desired")
    axs[5].set_title("Jc*F comparison")
    axs[5].legend()

    # Optional: label axes, add spacing
    for i in range(5):
        axs[i].set_xlabel("state")
        axs[i].set_ylabel("value")

    plt.tight_layout()
    plt.show()
    

    # print(f"\nBase orientation debug: {q_debug[3:6].flatten()}")
    # print(f"Velocity norm: {np.linalg.norm(qd_debug):.6f}")
    # print(f"Acceleration norm: {np.linalg.norm(qdd_debug):.6f}")
    
    # # Check constraint violations
    # try:
    #     ddq_check = fd(q_debug, qd_debug, u_debug, f_debug)
    #     print(f"Dynamics constraint error: {np.linalg.norm(ddq_check - qdd_debug):.6e}")
    # except:
    #     print("Could not evaluate dynamics constraint")
    
    # # Still create debug plots
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # ax.bar(['q norm', 'qd norm', 'qdd norm', 'u norm', 'f norm'],
    #        [np.linalg.norm(q_debug), np.linalg.norm(qd_debug), 
    #         np.linalg.norm(qdd_debug), np.linalg.norm(u_debug), 
    #         np.linalg.norm(f_debug)])
    # ax.set_ylabel('L2 Norm')
    # ax.set_title('Debug: Norms of Decision Variables')
    # plt.show()

exit()

fig, axs = plt.subplots(3, 3, figsize=(15, 8))  # Wider layout

# Flatten the 2D array of axes for easy indexing
axs = axs.flatten()

# First subplot: qdd comparison
axs[0].plot(qdd_optimal, label="qdd_optimized")
axs[0].plot(qdd, label="qdd_desired")
axs[0].set_title("qdd comparison")
axs[0].legend()

# Second subplot: q comparison
axs[1].plot(q_optimal, label="q_optimized")
axs[1].plot(q, label="q_desired")
axs[1].set_title("q comparison")
axs[1].legend()

# Third subplot: v comparison
axs[2].plot(qd_optimal, label="v_optimal")
axs[2].plot(qd, label="qd_desired")
axs[2].set_title("qd comparison")
axs[2].legend()

# Fourth subplot: tau comparison
axs[3].plot(u_optimal, label="tau_optimal")
axs[3].plot(u, label="u_desired")
axs[3].set_title("u comparison")
axs[3].legend()

# Fifth subplot: f comparison
axs[4].plot(f_optimal, label="f_optimal")
axs[4].plot(f, label="f_desired")
axs[4].set_title("f comparison")
axs[4].legend()

Jc_F_optimal= computeFullContactJacobians(q).T @ f
Jc_F_nominal_numerical = computeFullContactJacobians(q).T @ f
# 6 subplot: Jc comparison
axs[5].plot(Jc_F_optimal, label="Jc_F_optimal")
axs[5].plot(Jc_F_nominal_numerical, label="Jc_F_desired")
axs[5].set_title("Jc*F comparison")
axs[5].legend()

x_optimal = fk(q_optimal)
x_nominal_numerical = fk(q)
axs[6].plot(x_optimal, label="x_optimal")
axs[6].plot(x_nominal_numerical, label="x_desired")
axs[6].set_title("fk comparison")
axs[6].legend()

# Optional: label axes, add spacing
for i in range(6):
    axs[i].set_xlabel("Time step")
    axs[i].set_ylabel("Value")

plt.tight_layout()
plt.show()