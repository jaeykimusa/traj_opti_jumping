import numpy as np
from pathlib import Path
from sys import argv
import pinocchio 
from scipy.optimize import least_squares


def getDefaultStandState(model, data):
    # Sample a random configuration
    # q = pinocchio.randomConfiguration(model)
    # q = np.array([0.0, 0.0, 0.4, 0.4, 
    #               0.0, 0.0, 0.0, 
    #               0.0, 0.95, -1.75, 
    #               0.0, 0.95, -1.75, 
    #               0.0, 0.95, -1.75, 
    #               0.0, 0.95, -1.75], dtype=np.double)
    # print(f"q: {q.T}")
    base_xyz = np.array([0.0, 0.0, 0.3]) # init guess for z = 0.3 m
    base_quat = np.array([0.0, 0.0, 0.0])
    joints = np.array([0.0, 0.95, -1.75] * 4)
    q = np.concatenate([base_xyz, base_quat, joints])

    feet_names = ["FL_EE", "FR_EE", "RL_EE", "RR_EE"]
    feet_ids = [model.getFrameId(name) for name in feet_names]

    # Desired foot positions in world frame (z=0)
    target_positions = {
        "FL_EE": np.array([0.1934, 0.142, 0]),
        "FR_EE": np.array([0.1934, -0.142, 0]),
        "RL_EE": np.array([-0.1934, 0.142, 0]),
        "RR_EE": np.array([-0.1934, -0.142, 0]),
    }

    # ====================
    # 3. Inverse Kinematics Setup
    # ====================
    def pose_error(q):
        pinocchio.framesForwardKinematics(model, data, q)
        error = []
        for name, fid in zip(feet_names, feet_ids):
            err = data.oMf[fid].translation - target_positions[name]
            error.append(err)
        return np.concatenate(error)

    result = least_squares(
        fun=pose_error,
        x0=q,
        method="trf",
        max_nfev=100,
        verbose=0,
    )
    q_solved = result.x
    return q_solved

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

