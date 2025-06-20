# go2_stand.py

from pathlib import Path
from sys import argv

import pinocchio as pin
import numpy as np

from kinematics import *
from dynamics import *
from robot import * 
from math_utils import *
# from morphology import *

# =================================================================
#   PRIMARY GOAL IS TO COMPUTE GROUND REACTION FORCE AT EACH FOOT
# =================================================================

def printSize(matrix):
    print(matrix.shape)

# init
q = getDefaultStandState(model, data)
pin.framesForwardKinematics(model, data, q)

qd = np.zeros(model.nv)
qdd = np.zeros(model.nv)

Fc = computeFullContactForces(model, data, q, qd, qdd)



# i = Frame.FL_EE + 0
# print(i)
# print(model.frames[10])

# computeJointJacobians : computes ALL jacobians
# computeJointJacobian: computes only one joint jacobian
# computeFrameJacobian: computes only one frame jacobian
# getJointJacobian: If computeJointJacobians is already called, it gives the joint jacobian given by joint id j
# getFrameJacobian: If computeJointJacobians is already called, it gives the frame jacobian given by frame id f.

# pin.computeFrameJacobian(
# )




