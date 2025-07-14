# fd.py

from go2.robot.robot import *
from go2.robot.morphology import *
from go2.dynamics.fd import *
from go2.utils.math_utils import *

def computeFullContactJacobians(q):
    if isinstance(q, ca.SX):
        pinocchio.casadi.forwardKinematics(ad_model, ad_data, q)
        pinocchio.casadi.updateFramePlacements(ad_model, ad_data)
        Jc_FL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, 26, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jc_FR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, 58, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jc_RL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, 34, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jc_RR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, 66, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        full_Jc = ca.vertcat(Jc_FL, Jc_FR, Jc_RL, Jc_RR)
        Jc = full_Jc
    elif isinstance(q, ca.MX):
        cs_q = ca.SX.sym("q", NUM_Q, 1)
        cs_Jc = ca.SX.sym("Jc", NUM_U, NUM_Q)
        pinocchio.casadi.forwardKinematics(ad_model, ad_data, cs_q)
        pinocchio.casadi.updateFramePlacements(ad_model, ad_data)
        cs_Jc_FL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, 26, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc_FR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, 58, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc_RL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, 34, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc_RR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, 66, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc = ca.vertcat(cs_Jc_FL, cs_Jc_FR, cs_Jc_RL, cs_Jc_RR)
        cs_full_Jc_fn = ca.Function("cs_full_Jc_fn", [cs_q], [cs_Jc])
        Jc = cs_full_Jc_fn(q)
    else:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)
        J_list = []
        for frameId in EE_FRAME_IDS:
            J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
            J_list.append(J[:3, :])
        Jc = np.vstack(J_list)
    return Jc

def fd(q, v, tau_actuator, f):
    if isinstance(q, (ca.SX, ca.MX)):  # CasADi symbolic mode
        # Create symbolic variables
        cs_q = ca.SX.sym("q", NUM_Q, 1)
        cs_v = ca.SX.sym("qd", NUM_Q, 1)
        cs_tau_actuator = ca.SX.sym("tau", NUM_Q, 1)  # Fixed variable name
        cs_f = ca.SX.sym("f", NUM_F, 1)
        
        # Compute contact Jacobian symbolically
        cs_Jc = computeFullContactJacobians(cs_q)
        
        # Compute total generalized forces
        cs_tau_generalized = cs_tau_actuator + cs_Jc.T @ cs_f
        
        # Compute acceleration using ABA
        pinocchio.casadi.aba(ad_model, ad_data, cs_q, cs_v, cs_tau_generalized)
        a_ad = ad_data.ddq
        
        # Create function with correct inputs
        cs_aba_fn = ca.Function("create_aba_fn", [cs_q, cs_v, cs_tau_actuator, cs_f], [a_ad])
        
        # Evaluate at given values
        ddq = cs_aba_fn(q, v, tau_actuator, f)
       
    else: 
        Jc = computeFullContactJacobians(q)
        tau_generalized = tau_actuator + Jc.T @ f        
        pin.aba(model, data, q, v, tau_generalized)
        ddq = data.ddq

    return ddq

# Define computeGravity function that works with both symbolic and numerical inputs
def computeGravity(q):
    """
    Compute generalized gravity vector for the robot.
    Works with both CasADi symbolic and numerical inputs.
    
    Args:
        q: Configuration vector (18x1) - can be numerical or CasADi symbolic
        
    Returns:
        g: Generalized gravity vector (18x1)
    """
    if isinstance(q, (ca.SX, ca.MX)):  # CasADi symbolic mode
        # Create symbolic variable for configuration
        cs_q = ca.SX.sym("q", NUM_Q, 1)
        
        # Compute gravity using Pinocchio's CasADi interface
        pinocchio.casadi.computeGeneralizedGravity(ad_model, ad_data, cs_q)
        g_ad = ad_data.g
        
        # Create CasADi function
        cs_gravity_fn = ca.Function("compute_gravity_fn", [cs_q], [g_ad])
        
        # Evaluate at the given configuration
        g = cs_gravity_fn(q)
        
    else:  # Numerical mode
        # Compute gravity using standard Pinocchio
        g = pin.computeGeneralizedGravity(model, data, q)
        
    return g
