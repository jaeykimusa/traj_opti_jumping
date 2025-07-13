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
f = computeStandingContactForces(q)
u = id(q, qd, qdd, f)

x_d = fk(q)

opti = ca.Opti()

# Decision variables
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

# equality constraints
# opti.subject_to(qdd_opt == fd(q_opt, qd_opt, u_opt, f_opt))
# opti.subject_to(qdd_opt == np.zeros(NUM_Q).reshape(-1, 1))
# opti.subject_to(qd_opt == np.zeros(NUM_Q).reshape(-1, 1))
# opti.subject_to(x_d == fk(q_opt))
# opti.subject_to(u_opt == id(q_opt, qd_opt, qdd_opt, f_opt))

opti.subject_to(qdd_opt == np.zeros(NUM_Q).reshape(-1, 1))
opti.subject_to(qd_opt == np.zeros(NUM_Q).reshape(-1, 1))
opti.subject_to(x_d == fk(q_opt))
# inequality constraints
tau_max = 45.0  # Nm
joint_lower = np.array([-2.5, -2.5, -2.5] * 4)
joint_upper = np.array([2.5, 2.5, 2.5] * 4)
opti.subject_to(u_opt[6:18] >= -tau_max)
opti.subject_to(u_opt[6:18] <= tau_max)
opti.subject_to(q_opt[6:18] >= joint_lower.reshape(-1, 1))
opti.subject_to(q_opt[6:18] <= joint_upper.reshape(-1, 1))
for i in range(4):  # 4 feet
    foot_idx = i * 3
    fx = f_opt[foot_idx]
    fy = f_opt[foot_idx + 1] 
    fz = f_opt[foot_idx + 2]
    # Normal force must be positive
    opti.subject_to(fz >= 0.0)   # Minimum normal force
    opti.subject_to(fz <= 150.0)  # Maximum normal force
    opti.subject_to(fx <= MU * fz)
    opti.subject_to(fx >= -MU * fz)
    opti.subject_to(fy <= MU * fz)
    opti.subject_to(fy >= -MU * fz)

# ===== COST FUNCTION =====
cost = 0
Q_COST_WEIGHT = 1
QD_COST_WEIGHT = 1
QDD_COST_WEIGHT = 1
U_COST_WEIGHT = 0.01
F_COST_WEIGHT = 0.001

cost += Q_COST_WEIGHT * (q_opt - q_d).T @ (q_opt - q_d)
cost += QD_COST_WEIGHT * (qd_opt - qd_d).T @ (qd_opt - qd_d)
cost += QDD_COST_WEIGHT * (qdd_opt - qdd_d).T @ (qdd_opt - qdd_d)
cost += U_COST_WEIGHT * (u_opt - u_d).T @ (u_opt - u_d)
cost += F_COST_WEIGHT * (f_opt - f_d).T @ (f_opt - f_d)

opti.minimize(cost)

# Set parameter values
opti.set_value(q_d, q.reshape(-1, 1))
opti.set_value(qd_d, qd.reshape(-1, 1))
opti.set_value(qdd_d, qdd.reshape(-1, 1))
opti.set_value(u_d, u.reshape(-1, 1))
opti.set_value(f_d, f.reshape(-1, 1))

# Initial guess
opti.set_initial(q_opt, q.reshape(-1, 1))
opti.set_initial(qd_opt, qd.reshape(-1, 1))
opti.set_initial(qdd_opt, qdd.reshape(-1, 1))
opti.set_initial(u_opt, u.reshape(-1, 1))
opti.set_initial(f_opt, f.reshape(-1, 1))

# Solver options
opti.solver("ipopt", {"expand": True}, {"max_iter": 1000})

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
        print(f"    Friction utilization: {friction_ratio/MU*100:.1f}%")
    
    print(f"\nTotal vertical force - Optimal: {total_fz_opt:.2f} N")
    print(f"Expected weight: {12.0 * 9.81:.2f} N")

    visualize("test", q=q_optimal)
    
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
    

# exit()

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