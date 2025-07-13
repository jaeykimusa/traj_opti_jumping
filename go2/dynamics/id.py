# id.py

from go2.robot.robot import *
from go2.robot.morphology import *
from go2.dynamics.fd import *
from go2.utils.math_utils import *

def computeFullContactJacobians(q):
    if isinstance(q, ca.SX):
        Jc_FL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jc_FR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jc_RL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jc_RR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, q, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        full_Jc = ca.vertcat(Jc_FL, Jc_FR, Jc_RL, Jc_RR)
        Jc = full_Jc
    elif isinstance(q, ca.MX):
        cs_q = ca.SX.sym("q", NUM_Q, 1)
        cs_Jc = ca.SX.sym("Jc", NUM_U, NUM_Q)
        cs_Jc_FL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc_FR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc_RL = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc_RR = pinocchio.casadi.computeFrameJacobian(ad_model, ad_data, cs_q, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        cs_Jc = ca.vertcat(cs_Jc_FL, cs_Jc_FR, cs_Jc_RL, cs_Jc_RR)
        cs_full_Jc_fn = ca.Function("cs_full_Jc_fn", [cs_q], [cs_Jc])
        Jc = cs_full_Jc_fn(q)
    else:
        pin.computeJointJacobians(model, data, q)
        pin.framesForwardKinematics(model, data, q)
        J_list = []
        for frameId in EE_FRAME_IDS:
            J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
            J_list.append(J[:3, :])
        Jc = np.vstack(J_list)
    return Jc

def id(q, v, qdd, f):
    if isinstance(q, (ca.SX, ca.MX)):  # CasADi symbolic mode
        # Create symbolic variables
        cs_q = ca.SX.sym("q", NUM_Q, 1)
        cs_v = ca.SX.sym("qd", NUM_Q, 1)
        cs_qdd = ca.SX.sym("qdd", NUM_Q, 1)
        cs_f = ca.SX.sym("f", NUM_F, 1)
        
        # Compute contact Jacobian symbolically
        cs_Jc = computeFullContactJacobians(cs_q)
        
        # Compute total generalized forces
        cs_tau_rnea = pinocchio.casadi.rnea(ad_model, ad_data, cs_q, cs_v, cs_qdd) 
        cs_tau_actuator = cs_tau_rnea- cs_Jc.T @ cs_f
        
        # Create function with correct inputs
        cs_tau_fn = ca.Function("create_tau_fn", [cs_q, cs_v, cs_qdd, cs_f], [cs_tau_actuator])
        
        # Evaluate at given values
        tau_actuator = cs_tau_fn(q, v, qdd, f)
       
    else: 
        Jc = computeFullContactJacobians(q)
        tau_actuator = pin.rnea(model, data, q, v, qdd) - Jc.T @ f        

    return tau_actuator