import numpy as np
from pathlib import Path
from sys import argv
from scipy.optimize import least_squares

from go2.robot.robot import *
from go2.robot.morphology import *


# Alternative: If you want to optimize all values but ensure zero orientation
def getDefaultStandStateFullOptimization(model, data, foot_target_positions=None):
    # Default foot positions if not provided
    if foot_target_positions is None:
        foot_target_positions = {
            "LF_FOOT": np.array([0.1934, 0.142, 0]),
            "RF_FOOT": np.array([0.1934, -0.142, 0]),
            "LH_FOOT": np.array([-0.1934, 0.142, 0]),
            "RH_FOOT": np.array([-0.1934, -0.142, 0]),
        }
    
    feet_names = FOOT_NAMES
    feet_ids = EE_FRAME_IDS
    
    # Initial guess
    base_xyz = np.array([0.0, 0.0, 0.3])
    base_rpy = np.array([0.0, 0.0, 0.0])
    joints = np.array([0.0, 0.8, -1.6] * 4)
    
    q_init = np.concatenate([base_xyz, base_rpy, joints])
    
    # Bounds for optimization - allow tiny variations to satisfy optimizer
    lb = q_init.copy()
    ub = q_init.copy()
    
    # Base position bounds
    lb[:3] = np.array([-0.001, -0.001, 0.2])   # Very small x,y movement allowed
    ub[:3] = np.array([0.001, 0.001, 0.5])   
    
    # Base orientation bounds - very small tolerance
    tolerance = 1e-6
    lb[3:6] = np.array([-tolerance, -tolerance, -tolerance])
    ub[3:6] = np.array([tolerance, tolerance, tolerance])
    
    # Joint angle bounds
    joint_lb = np.array([-0.5, 0.5, -2.0] * 4)
    joint_ub = np.array([0.5, 1.5, -1.0] * 4)
    lb[6:] = joint_lb
    ub[6:] = joint_ub
    
    def pose_error(q):
        pinocchio.framesForwardKinematics(model, data, q)
        error = []
        
        # Foot position errors
        for name, fid in zip(feet_names, feet_ids):
            err = data.oMf[fid].translation - foot_target_positions[name]
            error.append(err)
        
        # Add strong penalty for orientation deviation
        orientation_penalty = q[3:6] * 10000.0
        error.append(orientation_penalty)
        
        # Add penalty for base x,y deviation
        base_xy_penalty = q[:2] * 1000.0
        error.append(base_xy_penalty)
        
        return np.concatenate(error)
    
    # Run optimization
    result = least_squares(
        fun=pose_error,
        x0=q_init,
        bounds=(lb, ub),
        method='trf',
        max_nfev=500,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        verbose=0,
    )
    
    # Force exact zeros for base x,y and orientation
    q_final = result.x.copy()
    q_final[0] = 0.0  # x
    q_final[1] = 0.0  # y
    q_final[3:6] = 0.0  # orientation
    
    return q_final

def printEEPositions(model, data):
    ee_names = ["FL_EE", "FR_EE", "RL_EE", "RR_EE"]
    ee_ids = [model.getFrameId(name) for name in ee_names]
    for ee_id in ee_ids:
        printEEPosition(model, data, ee_id)

def printEEPosition(model, data, ee_id):
    ee_name = model.frames[ee_id].name
    ee_pos = data.oMf[ee_id].translation
    adjusted_ee_pos = []
    zero_threshold = 1e-6 # 0.0000001
    for i in ee_pos:
        # Check if the absolute value of the coordinate is less than the threshold
        if abs(i) < zero_threshold:
            # If it's very close to zero, append a positive 0.0 to the adjusted list
            adjusted_ee_pos.append(0.0)
        else:
            # Otherwise, keep the original value
            adjusted_ee_pos.append(i)
    print("{}: {: .3f} {: .3f} {: .3f}".format(ee_name, *adjusted_ee_pos))

# center of mass of the robot
def getCoM(model, data, q):
    return pinocchio.centerOfMass(model, data, q)
def printCoM(model, data, q):
    print("Center of Mass: {: .3f} {: .3f} {: .3f}".format(*getCoM(model, data, q)))

# total mass of the robot
def getMass(model):
    return pinocchio.computeTotalMass(model)
def printMass(model):
    mass = getMass(model)
    print(f"Robot mass: {mass:.3f} kg")
def printExactMass(model):
    mass = getMass(model)
    print(f"Robot mass: {mass} kg")

# def computeContactJacobianFull():

