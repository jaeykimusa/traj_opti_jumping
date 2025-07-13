# dynamics.py
import pinocchio as pin
from go2.robot.robot import *
from go2.utils.math_utils import *
from go2.dynamics.id import *

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

def computeStandingContactForces(q):
    """
    Compute contact forces that achieve standing equilibrium.
    For standing: tau_motor + Jc^T * f = g
    Since tau_motor[0:6] = 0 (unactuated), we need: Jc[0:6,:]^T * f = g[0:6]
    """
    g = pin.computeGeneralizedGravity(model, data, q)
    Jc = computeFullContactJacobians(q)
    
    # Extract the part of Jacobian corresponding to floating base
    Jc_base = Jc[:, 0:6]  # Contact forces' effect on base
    
    # We need to solve: Jc_base^T * f = g[0:6]
    # This is an underdetermined system (12 unknowns, 6 equations)
    # We'll use least squares with additional constraints
    
    # Method 1: Pseudo-inverse (minimum norm solution)
    f = np.linalg.pinv(Jc_base.T) @ g[0:6]
    
    # Method 2: QP to ensure physical constraints
    # (Better approach - ensures positive normal forces, friction limits, etc.)
    
    return f

def computeContactJacobiansTimeVariation(q, qd):
    # J_FL = pin.getFrameJacobian(model, data, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_FL = pin.getFrameJacobianTimeVariation(model, data, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_FL = Jd_FL[:3, :]
    # J_FR = pin.getFrameJacobian(model, data, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_FR = pin.getFrameJacobianTimeVariation(model, data, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_FR = Jd_FL[:3, :]
    # J_RL = pin.getFrameJacobian(model, data, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_RL = pin.getFrameJacobianTimeVariation(model, data, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_RL = Jd_FL[:3, :]
    # J_RR = pin.getFrameJacobian(model, data, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_RR = pin.getFrameJacobianTimeVariation(model, data, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
    # Jd_RR = Jd_FL[:3, :]
    # Jd = np.vstack([Jd_FL, Jd_FR, Jd_RL, Jd_RR])

    pin.computeJointJacobiansTimeVariation(model, data, q, qd)
    Jd_FL_EE = pin.getFrameJacobianTimeVariation(model, data, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
    Jd_FL_EE = Jd_FL_EE[:3, :]
    Jd_FR_EE = pin.getFrameJacobianTimeVariation(model, data, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)
    Jd_FR_EE = Jd_FR_EE[:3, :]
    Jd_RL_EE = pin.getFrameJacobianTimeVariation(model, data, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)
    Jd_RL_EE = Jd_RL_EE[:3, :]
    Jd_RR_EE = pin.getFrameJacobianTimeVariation(model, data, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)
    Jd_RR_EE = Jd_RR_EE[:3, :]
    Jd = np.vstack([Jd_FL_EE, Jd_FR_EE, Jd_RL_EE, Jd_RR_EE])
    return Jd

def getMassInertiaMatrix(q):
    return pin.crba(model, data, q)

def getCoriolisGravity(q, qd, qdd):
    return pin.rnea(model, data, q, qd, qdd)