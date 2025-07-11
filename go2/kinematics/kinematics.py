import numpy as np
from pathlib import Path
from sys import argv
from scipy.optimize import least_squares

from go2.robot.robot import *
from go2.robot.morphology import *


def getDefaultStandState(model, data, foot_target_positions=None):
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
    
    # Initial guess - base at 0.3m height with zero orientation
    base_xyz = np.array([0.0, 0.0, 0.3])  # Initial guess for base position
    base_rpy = np.array([0.0, 0.0, 0.0])  # Initial orientation (roll, pitch, yaw)
    
    # Initial joint angles - typical standing configuration
    joints = np.array([0.0, 0.8, -1.6] * 4)  # Slightly bent legs
    
    q_init = np.concatenate([base_xyz, base_rpy, joints])
    
    # Bounds for optimization
    lb = q_init.copy()
    ub = q_init.copy()
    
    # Base position bounds
    lb[:3] = np.array([-0.1, -0.1, 0.2])   # x,y,z min
    ub[:3] = np.array([0.1, 0.1, 0.5])     # x,y,z max
    
    # Base orientation bounds (roll, pitch, yaw in radians)
    lb[3:6] = np.array([-0.2, -0.2, -0.2])  # Small allowed orientation variation
    ub[3:6] = np.array([0.2, 0.2, 0.2])
    
    # Joint angle bounds (adjust according to your robot's limits)
    joint_lb = np.array([-0.5, 0.5, -2.0] * 4)  # Min angles for each leg
    joint_ub = np.array([0.5, 1.5, -1.0] * 4)   # Max angles for each leg
    lb[6:] = joint_lb  # Note: changed from 7 to 6 because we have 6 base DOFs now
    ub[6:] = joint_ub
    
    def pose_error(q):
        pinocchio.framesForwardKinematics(model, data, q)
        error = []
        for name, fid in zip(feet_names, feet_ids):
            err = data.oMf[fid].translation - foot_target_positions[name]
            error.append(err)
        return np.concatenate(error)
    
    # Run optimization
    result = least_squares(
        fun=pose_error,
        x0=q_init,
        bounds=(lb, ub),
        method='trf',
        max_nfev=200,
        verbose=0,
    )
    
    return result.x

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

