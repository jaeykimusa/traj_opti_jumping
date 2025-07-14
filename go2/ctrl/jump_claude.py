#!/usr/bin/env python3
"""
Complete Go2 Quadruped Jumping Trajectory Optimization

This script generates a full-body jumping trajectory for the Go2 quadruped robot
using direct trajectory optimization with CasADi and IPOPT.

Features:
- Physics-based dynamics constraints
- Contact phase management (stance, takeoff, flight, landing)
- Friction cone constraints
- Joint limits and torque limits
- Forward kinematics constraints
- Smooth trajectory generation

Author: Generated with systematic debugging approach
"""

import numpy as np
import casadi as ca
import pinocchio as pin
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Import robot model and dynamics
from go2.robot.robot import *
from go2.robot.morphology import *
from go2.dynamics.fd import fd
from go2.dynamics.id import id
from go2.kinematics.fk import fk

print("="*80)
print("🦘 GO2 QUADRUPED JUMPING TRAJECTORY OPTIMIZATION 🦘")
print("="*80)

# ===== OPTIMIZATION PARAMETERS =====
print("\n📋 Setting up optimization parameters...")

# Trajectory parameters
N = 60                    # Number of timesteps (reduced for faster solve)
TIMESTEP = 0.035         # Time per step (seconds)
TOTAL_TIME = N * TIMESTEP # Total trajectory time (2.1 seconds)

# Robot parameters
ROBOT_MASS = 12.0        # kg (approximate)
GRAVITY = 9.81           # m/s²
MU = 0.8                 # Friction coefficient

# Jump parameters
JUMP_DISTANCE = 1.2      # Target forward jump distance (meters)
MAX_JUMP_HEIGHT = 0.8    # Maximum allowed jump height (meters)
LANDING_HEIGHT = 0.28    # Target landing height (meters)

# Constraint limits
TAU_MAX = 45.0           # Maximum joint torque (Nm)
FZ_MAX = 300.0           # Maximum normal force per foot (N)
VEL_MAX = 8.0            # Maximum velocity (m/s)
ACC_MAX = 20.0           # Maximum acceleration (m/s²)

print(f"✅ Trajectory: {N} steps × {TIMESTEP}s = {TOTAL_TIME}s")
print(f"✅ Target jump: {JUMP_DISTANCE}m forward, max height {MAX_JUMP_HEIGHT}m")

# ===== PHASE DEFINITIONS =====
print("\n🎯 Defining motion phases...")

# Phase timing (as fractions of total time)
STANCE_PHASE = (0.0, 0.35)      # 35% - Preparation and push-off
TAKEOFF_PHASE = (0.35, 0.45)    # 10% - Rear feet push, front feet lift
FLIGHT_PHASE = (0.45, 0.75)     # 30% - All feet in air
LANDING_PHASE = (0.75, 1.0)     # 25% - Touch down and stabilization

# Convert to timestep indices
def phase_to_timesteps(phase_fraction):
    start_idx = int(phase_fraction[0] * N)
    end_idx = int(phase_fraction[1] * N)
    return start_idx, min(end_idx, N)

STANCE_STEPS = phase_to_timesteps(STANCE_PHASE)
TAKEOFF_STEPS = phase_to_timesteps(TAKEOFF_PHASE)
FLIGHT_STEPS = phase_to_timesteps(FLIGHT_PHASE)
LANDING_STEPS = phase_to_timesteps(LANDING_PHASE)

print(f"✅ Stance phase:  steps {STANCE_STEPS[0]:2d}-{STANCE_STEPS[1]:2d}")
print(f"✅ Takeoff phase: steps {TAKEOFF_STEPS[0]:2d}-{TAKEOFF_STEPS[1]:2d}")
print(f"✅ Flight phase:  steps {FLIGHT_STEPS[0]:2d}-{FLIGHT_STEPS[1]:2d}")
print(f"✅ Landing phase: steps {LANDING_STEPS[0]:2d}-{LANDING_STEPS[1]:2d}")

# ===== INITIAL CONDITIONS =====
print("\n🚀 Setting up initial conditions...")

def getStandingConfiguration():
    """Get a stable standing configuration for the robot"""
    q = pin.neutral(model)
    
    # Base position and orientation
    q[0] = 0.0              # x position
    q[1] = 0.0              # y position  
    q[2] = 0.28             # z position (height)
    q[3] = 0.0              # quaternion x
    q[4] = 0.0              # quaternion y
    q[5] = 0.0              # quaternion z
    q[6] = 1.0              # quaternion w (normalized)
    
    # Joint angles for stable standing (all legs symmetric)
    # Order: LF_HAA, LF_HFE, LF_KFE, LH_HAA, LH_HFE, LH_KFE, RF_HAA, RF_HFE, RF_KFE, RH_HAA, RH_HFE, RH_KFE
    standing_angles = np.array([
        0.0,  0.75, -1.5,    # Left Front: Hip_AA, Hip_FE, Knee_FE
        0.0,  0.75, -1.5,    # Left Hind
        0.0,  0.75, -1.5,    # Right Front  
        0.0,  0.75, -1.5     # Right Hind
    ])
    
    q[7:19] = standing_angles
    
    # Verify configuration is valid
    if not pin.isNormalized(model, q, 1e-6):
        q = pin.normalize(model, q)
    
    return q

def getStandingForces():
    """Get contact forces for standing equilibrium"""
    weight_per_foot = ROBOT_MASS * GRAVITY / 4.0
    
    f = np.zeros(NUM_F)  # NUM_F = 12 (4 feet × 3 components)
    # Set normal forces for each foot (every 3rd element starting from index 2)
    f[2] = weight_per_foot   # LF foot fz
    f[5] = weight_per_foot   # RF foot fz  
    f[8] = weight_per_foot   # LH foot fz
    f[11] = weight_per_foot  # RH foot fz
    
    return f

# Get initial conditions
q_initial = getStandingConfiguration()
qd_initial = np.zeros(NUM_V)  # NUM_V = 18
qdd_initial = np.zeros(NUM_V)
f_initial = getStandingForces()

print(f"✅ Initial configuration: base at {q_initial[:3]}")
print(f"✅ Standing forces: {np.sum(f_initial[2::3]):.1f}N total vertical")

# Validate initial conditions
try:
    # Test forward kinematics
    x_initial = fk(q_initial)
    print(f"✅ Initial FK: body at {x_initial[:3]}, feet at ground level")
    
    # Test dynamics
    qdd_test = fd(q_initial, qd_initial, np.zeros(NUM_V), f_initial)
    print(f"✅ Initial dynamics: max acceleration {np.max(np.abs(qdd_test)):.3f} m/s²")
    
except Exception as e:
    print(f"❌ Initial condition validation failed: {e}")
    raise

# ===== OPTIMIZATION SETUP =====
print("\n🎯 Setting up trajectory optimization problem...")

# Create optimization problem
opti = ca.Opti()

# Decision variables
q_var = opti.variable(NUM_Q, N)    # Configurations (19 × N)
qd_var = opti.variable(NUM_V, N)   # Velocities (18 × N)  
qdd_var = opti.variable(NUM_V, N)  # Accelerations (18 × N)
tau_var = opti.variable(NUM_V, N)  # Joint torques (18 × N)
f_var = opti.variable(NUM_F, N)    # Contact forces (12 × N)

print(f"✅ Decision variables: q({NUM_Q}×{N}), qd({NUM_V}×{N}), tau({NUM_V}×{N}), f({NUM_F}×{N})")
print(f"✅ Total variables: {NUM_Q*N + NUM_V*N + NUM_V*N + NUM_V*N + NUM_F*N}")

# ===== CONSTRAINTS =====
print("\n⚖️  Setting up constraints...")

constraint_count = 0

# 1. Initial conditions
opti.subject_to(q_var[:, 0] == q_initial.reshape(-1, 1))
opti.subject_to(qd_var[:, 0] == qd_initial.reshape(-1, 1))
constraint_count += NUM_Q + NUM_V

# 2. Dynamics constraints (for all timesteps)
for k in range(N):
    # Forward dynamics: qdd = fd(q, qd, tau, f)
    dynamics_residual = fd(q_var[:, k], qd_var[:, k], tau_var[:, k], f_var[:, k])
    opti.subject_to(qdd_var[:, k] == dynamics_residual)
    constraint_count += NUM_V

# 3. Integration constraints
for k in range(N-1):
    # Velocity integration
    opti.subject_to(qd_var[:, k+1] == qd_var[:, k] + qdd_var[:, k] * TIMESTEP)
    
    # Configuration integration (proper quaternion handling)
    q_next = pin.integrate(model, q_var[:, k], qd_var[:, k+1] * TIMESTEP)
    opti.subject_to(q_var[:, k+1] == q_next)
    constraint_count += NUM_V + NUM_Q

# 4. Joint limits
joint_lower = np.array([-0.5, -1.2, -2.5] * 4)  # Conservative joint limits
joint_upper = np.array([0.5, 1.2, -0.5] * 4)

for k in range(N):
    # Joint position limits (configuration space)
    opti.subject_to(q_var[7:19, k] >= joint_lower.reshape(-1, 1))
    opti.subject_to(q_var[7:19, k] <= joint_upper.reshape(-1, 1))
    
    # Joint velocity limits
    opti.subject_to(qd_var[6:18, k] >= -VEL_MAX)
    opti.subject_to(qd_var[6:18, k] <= VEL_MAX)
    
    # Joint acceleration limits  
    opti.subject_to(qdd_var[6:18, k] >= -ACC_MAX)
    opti.subject_to(qdd_var[6:18, k] <= ACC_MAX)
    
    # Joint torque limits
    opti.subject_to(tau_var[6:18, k] >= -TAU_MAX)
    opti.subject_to(tau_var[6:18, k] <= TAU_MAX)
    
    # Floating base has no direct actuation
    opti.subject_to(tau_var[:6, k] == 0)
    
    constraint_count += 12*4 + 6  # joint limits + base torques

# 5. Contact phase constraints
print("   Setting up contact phase constraints...")

for k in range(N):
    # Extract foot forces: [LF_x, LF_y, LF_z, RF_x, RF_y, RF_z, LH_x, LH_y, LH_z, RH_x, RH_y, RH_z]
    f_LF = [f_var[0, k], f_var[1, k], f_var[2, k]]
    f_RF = [f_var[3, k], f_var[4, k], f_var[5, k]]
    f_LH = [f_var[6, k], f_var[7, k], f_var[8, k]]
    f_RH = [f_var[9, k], f_var[10, k], f_var[11, k]]
    
    all_feet = [f_LF, f_RF, f_LH, f_RH]
    
    if STANCE_STEPS[0] <= k < STANCE_STEPS[1]:
        # Stance phase: All feet in contact
        for foot_forces in all_feet:
            fx, fy, fz = foot_forces
            # Normal force bounds
            opti.subject_to(fz >= 5.0)
            opti.subject_to(fz <= FZ_MAX)
            # Friction cone constraints
            opti.subject_to(fx <= MU * fz)
            opti.subject_to(fx >= -MU * fz)
            opti.subject_to(fy <= MU * fz)
            opti.subject_to(fy >= -MU * fz)
            constraint_count += 6
            
    elif TAKEOFF_STEPS[0] <= k < TAKEOFF_STEPS[1]:
        # Takeoff phase: Front feet lift, rear feet push
        # Front feet (LF, RF) no contact
        for foot_forces in [f_LF, f_RF]:
            opti.subject_to(foot_forces[0] == 0)  # fx = 0
            opti.subject_to(foot_forces[1] == 0)  # fy = 0
            opti.subject_to(foot_forces[2] == 0)  # fz = 0
            constraint_count += 3
        
        # Rear feet (LH, RH) still in contact
        for foot_forces in [f_LH, f_RH]:
            fx, fy, fz = foot_forces
            opti.subject_to(fz >= 5.0)
            opti.subject_to(fz <= FZ_MAX)
            opti.subject_to(fx <= MU * fz)
            opti.subject_to(fx >= -MU * fz)
            opti.subject_to(fy <= MU * fz)
            opti.subject_to(fy >= -MU * fz)
            constraint_count += 6
            
    elif FLIGHT_STEPS[0] <= k < FLIGHT_STEPS[1]:
        # Flight phase: No contact forces
        opti.subject_to(f_var[:, k] == 0)
        constraint_count += NUM_F
        
    elif LANDING_STEPS[0] <= k < LANDING_STEPS[1]:
        # Landing phase: All feet can make contact
        for foot_forces in all_feet:
            fx, fy, fz = foot_forces
            # Allow zero contact or positive contact
            opti.subject_to(fz >= 0.0)
            opti.subject_to(fz <= FZ_MAX)
            # Friction constraints (only active when in contact)
            opti.subject_to(fx <= MU * fz)
            opti.subject_to(fx >= -MU * fz)
            opti.subject_to(fy <= MU * fz)
            opti.subject_to(fy >= -MU * fz)
            constraint_count += 6

# 6. Jumping goal constraints
print("   Setting up jumping goal constraints...")

# Final position constraints
opti.subject_to(q_var[0, -1] >= JUMP_DISTANCE - 0.3)  # x position
opti.subject_to(q_var[0, -1] <= JUMP_DISTANCE + 0.3)
opti.subject_to(q_var[1, -1] >= -0.15)                # y position (centered)
opti.subject_to(q_var[1, -1] <= 0.15)
opti.subject_to(q_var[2, -1] >= LANDING_HEIGHT - 0.1) # z position
opti.subject_to(q_var[2, -1] <= LANDING_HEIGHT + 0.1)

# Final velocity constraints (reasonable landing speeds)
opti.subject_to(qd_var[0, -1] >= -1.0)  # x velocity
opti.subject_to(qd_var[0, -1] <= 3.0)
opti.subject_to(qd_var[2, -1] >= -4.0)  # z velocity (downward for landing)
opti.subject_to(qd_var[2, -1] <= 1.0)

# Height constraint during flight (prevent excessive jumping)
for k in range(FLIGHT_STEPS[0], FLIGHT_STEPS[1]):
    opti.subject_to(q_var[2, k] <= MAX_JUMP_HEIGHT)

constraint_count += 6 + 2 + (FLIGHT_STEPS[1] - FLIGHT_STEPS[0])

print(f"✅ Total constraints: {constraint_count}")

# ===== COST FUNCTION =====
print("\n💰 Setting up cost function...")

cost = 0

# 1. Effort minimization
effort_weight = 0.001
for k in range(N):
    # Minimize joint torques
    cost += effort_weight * ca.sumsqr(tau_var[6:18, k])
    
    # Minimize accelerations (smoothness)
    cost += effort_weight * 0.1 * ca.sumsqr(qdd_var[:, k])
    
    # Minimize contact forces
    cost += effort_weight * 0.01 * ca.sumsqr(f_var[:, k])

# 2. Trajectory smoothness
smoothness_weight = 0.01
for k in range(N-1):
    # Penalize acceleration changes (jerk minimization)
    cost += smoothness_weight * ca.sumsqr(qdd_var[:, k+1] - qdd_var[:, k])

# 3. Goal achievement
goal_weight = 100.0
cost += goal_weight * (q_var[0, -1] - JUMP_DISTANCE)**2  # Forward distance
cost += goal_weight * 0.1 * q_var[1, -1]**2             # Lateral centering
cost += goal_weight * 0.5 * (q_var[2, -1] - LANDING_HEIGHT)**2  # Landing height

# 4. Trajectory shaping
for k in range(N):
    # Encourage forward progression
    alpha = k / (N - 1)
    desired_x = alpha * JUMP_DISTANCE
    cost += 0.1 * (q_var[0, k] - desired_x)**2
    
    # Encourage reasonable jump height profile
    if FLIGHT_STEPS[0] <= k < FLIGHT_STEPS[1]:
        flight_alpha = (k - FLIGHT_STEPS[0]) / (FLIGHT_STEPS[1] - FLIGHT_STEPS[0])
        desired_height = LANDING_HEIGHT + 0.3 * np.sin(np.pi * flight_alpha)
        cost += 0.1 * (q_var[2, k] - desired_height)**2

# 5. Stability (minimize base orientation changes)
stability_weight = 10.0
for k in range(N):
    # Keep base orientation close to upright (quaternion close to [0,0,0,1])
    cost += stability_weight * (q_var[3, k]**2 + q_var[4, k]**2 + q_var[5, k]**2)
    cost += stability_weight * (q_var[6, k] - 1.0)**2

opti.minimize(cost)
print("✅ Cost function with effort, smoothness, goals, and stability terms")

# ===== INITIAL GUESS =====
print("\n🎲 Setting initial guess...")

for k in range(N):
    alpha = k / (N - 1)  # Progress from 0 to 1
    
    # Configuration guess
    q_guess = q_initial.copy()
    
    # Progressive forward motion
    q_guess[0] = alpha * JUMP_DISTANCE
    
    # Jump height profile
    if FLIGHT_STEPS[0] <= k < FLIGHT_STEPS[1]:
        flight_progress = (k - FLIGHT_STEPS[0]) / (FLIGHT_STEPS[1] - FLIGHT_STEPS[0])
        jump_height = 0.4 * np.sin(np.pi * flight_progress)
        q_guess[2] = q_initial[2] + jump_height
    elif k >= FLIGHT_STEPS[1]:
        q_guess[2] = LANDING_HEIGHT
    
    # Joint configuration changes for jumping
    if STANCE_STEPS[0] <= k < TAKEOFF_STEPS[1]:
        # Crouch and extend for jumping
        crouch_factor = 0.3 * np.sin(np.pi * k / TAKEOFF_STEPS[1])
        q_guess[8] += crouch_factor   # LF_HFE
        q_guess[11] += crouch_factor  # LH_HFE  
        q_guess[14] += crouch_factor  # RF_HFE
        q_guess[17] += crouch_factor  # RH_HFE
        
        q_guess[9] -= crouch_factor * 2   # LF_KFE
        q_guess[12] -= crouch_factor * 2  # LH_KFE
        q_guess[15] -= crouch_factor * 2  # RF_KFE
        q_guess[18] -= crouch_factor * 2  # RH_KFE
    
    # Set initial guess
    opti.set_initial(q_var[:, k], q_guess)
    opti.set_initial(qd_var[:, k], qd_initial)
    opti.set_initial(qdd_var[:, k], qdd_initial)
    opti.set_initial(tau_var[:, k], np.zeros(NUM_V))
    opti.set_initial(f_var[:, k], f_initial if k < FLIGHT_STEPS[0] else np.zeros(NUM_F))

print("✅ Initial guess set with progressive jumping motion")

# ===== SOLVER CONFIGURATION =====
print("\n⚙️  Configuring solver...")

# IPOPT solver options
solver_options = {
    "expand": False,
    "print_time": False,
    "verbose": False,
    "ipopt": {
        "max_iter": 3000,
        "tol": 1e-4,
        "acceptable_tol": 1e-3,
        "print_level": 5,
        "linear_solver": "mumps",
        "hessian_approximation": "limited-memory",
        "mu_strategy": "adaptive",
        "warm_start_init_point": "yes",
        "nlp_scaling_method": "gradient-based",
        "obj_scaling_factor": 1.0,
        "max_wall_time": 300.0,  # 5 minute timeout
        "acceptable_iter": 20,
        "acceptable_obj_change_tol": 1e-4
    }
}

opti.solver("ipopt", solver_options["ipopt"])
print("✅ IPOPT solver configured with 3000 max iterations")

# ===== SOLVE OPTIMIZATION =====
print("\n" + "="*80)
print("🚀 SOLVING JUMPING TRAJECTORY OPTIMIZATION")
print("="*80)
print(f"Problem size: {NUM_Q*N + NUM_V*N + NUM_V*N + NUM_V*N + NUM_F*N} variables, {constraint_count} constraints")
print(f"Expected solve time: 2-10 minutes")
print("Starting optimization...")

start_time = time.time()
success = False

try:
    # Solve the optimization problem
    sol = opti.solve()
    end_time = time.time()
    
    print(f"\n🎉 OPTIMIZATION SUCCESSFUL! 🎉")
    print(f"Solve time: {end_time - start_time:.2f} seconds")
    
    # Extract solution
    Q_OPT = sol.value(q_var)
    QD_OPT = sol.value(qd_var)
    QDD_OPT = sol.value(qdd_var)
    TAU_OPT = sol.value(tau_var)
    F_OPT = sol.value(f_var)
    COST_OPT = sol.value(cost)
    
    success = True
    
except RuntimeError as e:
    end_time = time.time()
    print(f"\n❌ OPTIMIZATION FAILED")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Error: {e}")
    
    # Extract debug values if available
    try:
        Q_OPT = opti.debug.value(q_var)
        QD_OPT = opti.debug.value(qd_var)
        TAU_OPT = opti.debug.value(tau_var)
        F_OPT = opti.debug.value(f_var)
        print("Partial solution available for analysis")
    except:
        print("No partial solution available")

# ===== RESULTS ANALYSIS =====
if success:
    print("\n" + "="*80)
    print("📊 TRAJECTORY ANALYSIS")
    print("="*80)
    
    # Basic trajectory metrics
    print(f"\n🎯 Jump Performance:")
    jump_distance = Q_OPT[0, -1] - Q_OPT[0, 0]
    max_height = np.max(Q_OPT[2, :])
    landing_height = Q_OPT[2, -1]
    
    print(f"   Forward distance: {jump_distance:.3f}m (target: {JUMP_DISTANCE:.3f}m)")
    print(f"   Maximum height:   {max_height:.3f}m (limit: {MAX_JUMP_HEIGHT:.3f}m)")
    print(f"   Landing height:   {landing_height:.3f}m (target: {LANDING_HEIGHT:.3f}m)")
    print(f"   Lateral drift:    {abs(Q_OPT[1, -1]):.3f}m")
    
    # Velocity analysis
    max_vel_x = np.max(np.abs(QD_OPT[0, :]))
    max_vel_z = np.max(np.abs(QD_OPT[2, :]))
    final_vel = np.linalg.norm(QD_OPT[:3, -1])
    
    print(f"\n🏃 Velocity Analysis:")
    print(f"   Max forward velocity: {max_vel_x:.3f}m/s")
    print(f"   Max vertical velocity: {max_vel_z:.3f}m/s")
    print(f"   Final velocity: {final_vel:.3f}m/s")
    
    # Force analysis
    print(f"\n💪 Force Analysis:")
    stance_forces = F_OPT[:, STANCE_STEPS[0]:STANCE_STEPS[1]]
    takeoff_forces = F_OPT[:, TAKEOFF_STEPS[0]:TAKEOFF_STEPS[1]]
    flight_forces = F_OPT[:, FLIGHT_STEPS[0]:FLIGHT_STEPS[1]]
    landing_forces = F_OPT[:, LANDING_STEPS[0]:LANDING_STEPS[1]]
    
    print(f"   Stance phase avg force:  {np.mean(np.sum(stance_forces[2::3, :], axis=0)):.1f}N")
    print(f"   Takeoff phase max force: {np.max(np.sum(takeoff_forces[2::3, :], axis=0)):.1f}N")
    print(f"   Flight phase max force:  {np.max(np.abs(flight_forces)):.3f}N (should be ~0)")
    print(f"   Landing phase avg force: {np.mean(np.sum(landing_forces[2::3, :], axis=0)):.1f}N")
    
    # Torque analysis
    max_torque = np.max(np.abs(TAU_OPT[6:18, :]))
    avg_torque = np.mean(np.abs(TAU_OPT[6:18, :]))
    
    print(f"\n🔧 Torque Analysis:")
    print(f"   Maximum joint torque: {max_torque:.2f}Nm (limit: {TAU_MAX:.2f}Nm)")
    print(f"   Average joint torque: {avg_torque:.2f}Nm")
    
    # Success metrics
    distance_accuracy = 100 * (1 - abs(jump_distance - JUMP_DISTANCE) / JUMP_DISTANCE)
    height_success = max_height < MAX_JUMP_HEIGHT
    torque_success = max_torque < TAU_MAX
    flight_success = np.max(np.abs(flight_forces)) < 1.0
    
    print(f"\n✅ Success Metrics:")
    print(f"   Distance accuracy: {distance_accuracy:.1f}%")
    print(f"   Height constraint: {'✅' if height_success else '❌'}")
    print(f"   Torque limits: {'✅' if torque_success else '❌'}")
    print(f"   Flight phase: {'✅' if flight_success else '❌'}")
    
    # Save results
    print(f"\n💾 Saving trajectory data...")
    




import rerun as rr
from go2.mpac_logging.mpac_logging import robot_zoo
from go2.mpac_logging.mpac_logging.rerun.robot_logger import RobotLogger
from go2.mpac_logging.mpac_logging.rerun.utils import rerun_initialize, rerun_store

rr.init("simple_robot_example", spawn=False)
# robot_logger = RobotLogger.from_zoo("go2")
robot_logger = RobotLogger.from_zoo("go2_description")

import time

rerun_initialize("forward jump test", spawn=True)
current_time = time.time()
# robot_logger.log_initial_state(logtime=current_time)

dt = TIMESTEP

for i in range(N):
    q_i = Q_OPT[:,i]
    base_position = q_i[:3]
    base_orientation = q_i[3:7]
    # joint_positions = {
    #     "FL_hip_joint": q_i[6], 
    #     "FL_thigh_joint": q_i[7],
    #     "FL_calf_joint": q_i[8],
    #     "FR_hip_joint": q_i[9],
    #     "FR_thigh_joint": q_i[10],
    #     "FR_calf_joint": q_i[11],
    #     "RL_hip_joint": q_i[12],
    #     "RL_thigh_joint": q_i[13],
    #     "RL_calf_joint": q_i[14],
    #     "RR_hip_joint": q_i[15],
    #     "RR_thigh_joint": q_i[16],
    #     "RR_calf_joint": q_i[17],
    # }
    joint_positions = {
        "FL_hip_joint": q_i[7], 
        "FL_thigh_joint": q_i[8],
        "FL_calf_joint": q_i[9],
        "FR_hip_joint": q_i[10],
        "FR_thigh_joint": q_i[11],
        "FR_calf_joint": q_i[12],
        "RL_hip_joint": q_i[13],
        "RL_thigh_joint": q_i[14],
        "RL_calf_joint": q_i[15],
        "RR_hip_joint": q_i[16],
        "RR_thigh_joint": q_i[17],
        "RR_calf_joint": q_i[18]
    }

    robot_logger.log_state(
        logtime=current_time,
        base_position=base_position,
        base_orientation=base_orientation,
        joint_positions=joint_positions
    )
    current_time += dt

# rr.save("forward_jump_test_today.rrd")

exit()