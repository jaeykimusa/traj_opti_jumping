# fd.py

import numpy as np
import casadi as ca
from go2.robot.robot import *
import pinocchio as pin
from go2.dynamics.dynamics import *
from go2.utils.math_utils import *


# symbolic
from pinocchio.casadi import crba, rnea#, computeJointPlacement, computeFramePlacements

# def fd(q, qd, qdd=None, u=None, f=None):
#     if qdd is None:
#         qdd = ca.SX.zeros(NUM_Q)
    
#     # # mass matrix
#     # M = pin.crba(model_casadi, data_casadi, convert_3Drot_to_quat(q)) 
#     # # nonLinearTerms 
#     # b = pin.rnea(model_casadi, data_casadi, q, qd, qdd)

#     q_sx = ca.SX.sym("q", NUM_Q)
#     qd_sx = ca.SX.sym("qd", NUM_Q)
#     qdd_sx = ca.SX.sym("qdd", NUM_Q)
#     M_fun = ca.Function("M_fun", [q_sx], [pin.crba(model_casadi, data_casadi, q_sx)])
#     b_fun = ca.Function("b_fun", [q_sx, qd_sx, qdd_sx], [pin.rnea(model_casadi, data_casadi, q_sx, qd_sx, qdd_sx)])
#     q_full = convert_3Drot_to_quat(q)
#     M = M_fun(q_full)
#     b = b_fun(q_full, qd, qdd)

#     # actuator selection mattrix B
#     B = ca.MX.zeros(NUM_Q, NUM_U)
#     B[6:, :] = ca.MX.eye(NUM_U)

#     # contact
#     Jc_FL = pin.computeFrameJacobian(model_casadi, data_casadi, q, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)
#     Jc_FR = pin.computeFrameJacobian(model_casadi, data_casadi, q, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)
#     Jc_RL = pin.computeFrameJacobian(model_casadi, data_casadi, q, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)
#     Jc_RR = pin.computeFrameJacobian(model_casadi, data_casadi, q, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)
#     Jc_FL = Jc_FL[:3, :]
#     Jc_FR = Jc_FL[:3, :]
#     Jc_RL = Jc_FL[:3, :]
#     Jc_RR = Jc_FL[:3, :]
#     Jc = np.vstack([Jc_FL, Jc_FR, Jc_RL, Jc_RR])

#     # solution to the eom when the qdd is zero
#     u0 = B @ u + Jc.T @ f - b
#     qdd = ca.solve(M, u0)

#     return qdd

#     # # s = symbolic
#     # fds = ca.Function("fd",
#     #                     [q, qd, u, f],
#     #                     [qdd],
#     #                     ["q", "qd", "u", "f"],
#     #                     ["qdd"])
#     # qdd_i = fds(q, qd, u, f)
#     # return qdd_i


# def build_fd_function():
#     q = ca.SX.sym("q", NUM_Q)
#     qd = ca.SX.sym("qd", NUM_Q)
#     u = ca.SX.sym("u", NUM_U)
#     f = ca.SX.sym("f", NUM_F)

#     q_full = convert_3Drot_to_quat(q)

#     M = pin.crba(model_casadi, data_casadi, q_full)
#     b = pin.rnea(model_casadi, data_casadi, q_full, qd, ca.SX.zeros(NUM_Q))

#     B = ca.SX.zeros(NUM_Q, NUM_U)
#     B[6:, :] = ca.SX.eye(NUM_U)

#     pin.computeJointPlacement(model_casadi, data_casadi, q_full)
#     pin.computeFramePlacements(model_casadi, data_casadi)

#     Jc_FL = pin.computeFrameJacobian(model_casadi, data_casadi, q_full, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
#     Jc_FR = pin.computeFrameJacobian(model_casadi, data_casadi, q_full, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
#     Jc_RL = pin.computeFrameJacobian(model_casadi, data_casadi, q_full, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
#     Jc_RR = pin.computeFrameJacobian(model_casadi, data_casadi, q_full, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
#     Jc = ca.vertcat(Jc_FL, Jc_FR, Jc_RL, Jc_RR)

#     tau = B @ u + Jc.T @ f - b
#     qdd = ca.solve(M, tau)

#     return ca.Function("fd_fun", [q, qd, u, f], [qdd])

def fd(q, qd, qdd=None, u=None, f=None):
    if qdd is None:
        qdd = np.zeros(NUM_Q)

    # Convert quaternion
    q_full = convert_3Drot_to_quat(q)  # returns np.array

    # --- Compute M(q) and b(q, qd, qdd) as NumPy ---
    M_np = pin.crba(model, data, q_full)
    b_np = pin.rnea(model, data, q_full, qd, qdd)

    # --- Convert to symbolic MX for downstream math ---
    M = ca.MX(M_np)
    b = ca.MX(b_np)

    # Actuation matrix B
    B = ca.MX.zeros(NUM_Q, NUM_U)
    B[6:, :] = ca.MX.eye(NUM_U)

    # Contact Jacobians using symbolic model
    q_sym = ca.MX(q_full)
    # computeJointPlacement(data_casadi, q_sym)
    # computeFramePlacements(data_casadi)

    pin.computeJointPlacement(model_casadi, data_casadi, q_sym)
    pin.computeFramePlacements(model_casadi, data_casadi)

    
    Jc_FL = pin.computeFrameJacobian(model_casadi, data_casadi, q_sym, Frame.FL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
    Jc_FR = pin.computeFrameJacobian(model_casadi, data_casadi, q_sym, Frame.FR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
    Jc_RL = pin.computeFrameJacobian(model_casadi, data_casadi, q_sym, Frame.RL_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]
    Jc_RR = pin.computeFrameJacobian(model_casadi, data_casadi, q_sym, Frame.RR_EE, pin.LOCAL_WORLD_ALIGNED)[:3, :]

    Jc = ca.vertcat(Jc_FL, Jc_FR, Jc_RL, Jc_RR)

    # Dynamics equation
    tau = B @ u + Jc.T @ f - b
    qdd_out = ca.solve(M, tau)

    return qdd_out 




