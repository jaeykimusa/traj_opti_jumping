# dynamics.py
import pinocchio as pin
from robot import *
from math_utils import *

def computeJointTorques(model, data, q, qd, qdd):
    return pin.rnea(model, data, q, qd, qdd).reshape(-1,1) # 18 x 1 

def computeContactJacobian(model, data, q):
    # size of 12 x 18
    # TODO: determine the number of contacts accurately.
    Jc_FL = pin.computeFrameJacobian(model, data, q, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jc_FL = pin.computeJointJacobian(model, data, q, model.frames[10], pin.LOCAL_WORLD_ALIGNED)
    Jc_FR = pin.computeFrameJacobian(model, data, q, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)
    Jc_RL = pin.computeFrameJacobian(model, data, q, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)
    Jc_RR = pin.computeFrameJacobian(model, data, q, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)

    Jc_FL = Jc_FL[:3, :]
    Jc_FR = Jc_FL[:3, :]
    Jc_RL = Jc_FL[:3, :]
    Jc_RR = Jc_FL[:3, :]

    Jc = np.vstack([Jc_FL, Jc_FR, Jc_RL, Jc_RR])
    return Jc

def computeFullContactForces(model, data, q, qd, qdd):
    # Pinocchio Doc: The desired joint torques stored in data.tau.
    tau = computeJointTorques(model, data, q, qd, qdd) # 18 x 1
    Jc = computeContactJacobian(model, data, q)
    Jc_T = Jc.T
    Jc_plus = getPseudoInverse(Jc_T)

    Fc = []
    Fc = (Jc_plus @ tau).flatten()
    # F_c_FL = F_c[:3]

    return Fc
    
def computeContactForces(model, data, q, qd, qdd):
    # Pinocchio Doc: The desired joint torques stored in data.tau.
    tau = computeJointTorques(model, data, q, qd, qdd) # 18 x 1
    Jc = computeContactJacobian(model, data, q)
    Jc_T = Jc.T
    Jc_plus = getPseudoInverse(Jc_T)

    Fc = []
    Fc = (Jc_plus @ tau).flatten()
    
    Fc_FL = Fc[0:3]
    Fc_FR = Fc[3:6]
    Fc_RL = Fc[6:9]
    Fc_RR = Fc[9:12]

    return Fc_FL, Fc_FR, Fc_RL, Fc_RR

def printContactForces(*args):
    if len(args) == 1:
        Fc = args[0].flatten()
        Fc_FL = Fc[0:3]
        Fc_FR = Fc[3:6]
        Fc_RL = Fc[6:9]
        Fc_RR = Fc[9:12]
        print(f"Force at FL_EE: {Fc_FL[0]:.3f} {Fc_FL[1]:.3f} {Fc_FL[2]:.3f}")
        print(f"Force at FR_EE: {Fc_FR[0]:.3f} {Fc_FR[1]:.3f} {Fc_FR[2]:.3f}")
        print(f"Force at RL_EE: {Fc_RL[0]:.3f} {Fc_RL[1]:.3f} {Fc_RL[2]:.3f}")
        print(f"Force at RR_EE: {Fc_RR[0]:.3f} {Fc_RR[1]:.3f} {Fc_RR[2]:.3f}")
    elif len(args) == 4:
        Fc_FL = args[0].flatten()
        Fc_FR = args[1].flatten()
        Fc_RL = args[2].flatten()
        Fc_RR = args[3].flatten()
        print(f"Force at FL_EE: {Fc_FL[0]:.3f} {Fc_FL[1]:.3f} {Fc_FL[2]:.3f}")
        print(f"Force at FR_EE: {Fc_FR[0]:.3f} {Fc_FR[1]:.3f} {Fc_FR[2]:.3f}")
        print(f"Force at RL_EE: {Fc_RL[0]:.3f} {Fc_RL[1]:.3f} {Fc_RL[2]:.3f}")
        print(f"Force at RR_EE: {Fc_RR[0]:.3f} {Fc_RR[1]:.3f} {Fc_RR[2]:.3f}")
    else:
        raise ValueError("Expected 1 or 4 arguments.")
