#!/usr/bin/env python3

import argparse

from trajopt_logging import get_logger
from mpac_logging.rerun.utils import rerun_initialize
from mpac_logging.rerun.robot_logger import RobotLogger

from robot_descriptions.loaders.pinocchio import load_robot_description

import pinocchio as pin
import pinocchio.casadi as cpin
import casadi

from dataclasses import dataclass, fields
from typing import Union, List

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R



@dataclass
class ForwardKinematicsResult:
  com_pos: Union[np.ndarray, casadi.SX]
  lf_pos: Union[np.ndarray, casadi.SX]
  lh_pos: Union[np.ndarray, casadi.SX]
  rf_pos: Union[np.ndarray, casadi.SX]
  rh_pos: Union[np.ndarray, casadi.SX]
  lf_knee_pos: Union[np.ndarray, casadi.SX]
  lh_knee_pos: Union[np.ndarray, casadi.SX]
  rf_knee_pos: Union[np.ndarray, casadi.SX]
  rh_knee_pos: Union[np.ndarray, casadi.SX]


@dataclass
class JacobianResult:
  com_jac: Union[np.ndarray, casadi.SX]
  lf_jac: Union[np.ndarray, casadi.SX]
  lh_jac: Union[np.ndarray, casadi.SX]
  rf_jac: Union[np.ndarray, casadi.SX]
  rh_jac: Union[np.ndarray, casadi.SX]


@dataclass
class JacobianDerivativeResult:
  com_jac_dot: Union[np.ndarray, casadi.SX]
  lf_jac_dot: Union[np.ndarray, casadi.SX]
  lh_jac_dot: Union[np.ndarray, casadi.SX]
  rf_jac_dot: Union[np.ndarray, casadi.SX]
  rh_jac_dot: Union[np.ndarray, casadi.SX]


def forward_kinematics(model, data, q: np.ndarray) -> ForwardKinematicsResult:
  pin.framesForwardKinematics(model, data, q)
  pin.updateFramePlacements(model, data)
  return ForwardKinematicsResult(
    com_pos=data.oMf[model.getFrameId("base")].translation,
    lf_pos=data.oMf[model.getFrameId("FL_foot")].translation,
    lh_pos=data.oMf[model.getFrameId("RL_foot")].translation,
    rf_pos=data.oMf[model.getFrameId("FR_foot")].translation,
    rh_pos=data.oMf[model.getFrameId("RR_foot")].translation,
    lf_knee_pos=data.oMf[model.getFrameId("FL_calf")].translation,
    lh_knee_pos=data.oMf[model.getFrameId("RL_calf")].translation,
    rf_knee_pos=data.oMf[model.getFrameId("FR_calf")].translation,
    rh_knee_pos=data.oMf[model.getFrameId("RR_calf")].translation,
  )

def compute_jacobians(model, data, q: np.ndarray) -> JacobianResult:
  pin.computeJointJacobians(model, data, q)
  pin.framesForwardKinematics(model, data, q)
  return JacobianResult(
    com_jac=pin.getFrameJacobian(model, data, model.getFrameId("base"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    lf_jac=pin.getFrameJacobian(model, data, model.getFrameId("FL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    lh_jac=pin.getFrameJacobian(model, data, model.getFrameId("RL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    rf_jac=pin.getFrameJacobian(model, data, model.getFrameId("FR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    rh_jac=pin.getFrameJacobian(model, data, model.getFrameId("RR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
  )


def compute_jacobian_derivatives(model, data, q: np.ndarray, v: np.ndarray) -> JacobianDerivativeResult:
  pin.computeJointJacobiansTimeVariation(model, data, q, v)
  # pin.framesForwardKinematics(model, data, q)
  
  return JacobianDerivativeResult(
    com_jac_dot =pin.getFrameJacobianTimeVariation(model, data, model.getFrameId("base"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_lf_dot = pin.getFrameJacobianTimeVariation(model, data, model.getFrameId("FL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_lh_dot = pin.getFrameJacobianTimeVariation(model, data, model.getFrameId("RL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_rf_dot = pin.getFrameJacobianTimeVariation(model, data, model.getFrameId("FR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_rh_dot = pin.getFrameJacobianTimeVariation(model, data, model.getFrameId("RR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
  )


def forward_dynamics(model, data, q: np.ndarray, v: np.ndarray, tau: np.ndarray, f: np.ndarray) -> np.ndarray:
  jacres = compute_jacobians(model, data, q)
  jacobians: List[np.ndarray] = [jacres.lf_jac, jacres.lh_jac, jacres.rf_jac, jacres.rh_jac]
  jacobians_stacked = np.concatenate(jacobians, axis=0)
  tau_total = tau + np.transpose(jacobians_stacked) @ f
  pin.aba(model, data, q, v, tau_total)
  return data.ddq


def forward_kinematics_casadi(cmodel, cdata, q: casadi.SX) -> ForwardKinematicsResult:
  cpin.framesForwardKinematics(cmodel, cdata, q)
  cpin.updateFramePlacements(cmodel, cdata)
  return ForwardKinematicsResult(
    com_pos=cdata.oMf[cmodel.getFrameId("base")].translation,
    lf_pos=cdata.oMf[cmodel.getFrameId("FL_foot")].translation,
    lh_pos=cdata.oMf[cmodel.getFrameId("RL_foot")].translation,
    rf_pos=cdata.oMf[cmodel.getFrameId("FR_foot")].translation,
    rh_pos=cdata.oMf[cmodel.getFrameId("RR_foot")].translation,
    lf_knee_pos=cdata.oMf[cmodel.getFrameId("FL_calf")].translation,
    lh_knee_pos=cdata.oMf[cmodel.getFrameId("RL_calf")].translation,
    rf_knee_pos=cdata.oMf[cmodel.getFrameId("FR_calf")].translation,
    rh_knee_pos=cdata.oMf[cmodel.getFrameId("RR_calf")].translation,
  )


def compute_jacobians_casadi(cmodel, cdata, q: casadi.SX) -> JacobianResult:
  cpin.computeJointJacobians(cmodel, cdata, q)
  cpin.framesForwardKinematics(cmodel, cdata, q)
  return JacobianResult(
    com_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("base"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    lf_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("FL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    lh_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("RL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    rf_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("FR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    rh_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("RR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
  )


def compute_jacobian_derivatives_casadi(cmodel, cdata, q: casadi.SX, v: casadi.SX) -> JacobianDerivativeResult:
  cpin.computeJointJacobiansTimeVariation(cmodel, cdata, q, v)
  # cpin.framesForwardKinematics(cmodel, cdata, q)
  return JacobianDerivativeResult(
    com_jac_dot = cpin.getFrameJacobianTimeVariation(cmodel, cdata, cmodel.getFrameId("base"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_lf_dot = cpin.getFrameJacobianTimeVariation(cmodel, cdata, cmodel.getFrameId("FL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_lh_dot = cpin.getFrameJacobianTimeVariation(cmodel, cdata, cmodel.getFrameId("RL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_rf_dot = cpin.getFrameJacobianTimeVariation(cmodel, cdata, cmodel.getFrameId("FR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
    jac_rh_dot = cpin.getFrameJacobianTimeVariation(cmodel, cdata, cmodel.getFrameId("RR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
  )


def forward_dynamics_casadi(cmodel, cdata, q: casadi.SX, v: casadi.SX, tau: casadi.SX, f: casadi.SX) -> casadi.SX:
  jacres = compute_jacobians_casadi(cmodel, cdata, q)
  jacobians: List[casadi.SX] = [jacres.lf_jac, jacres.lh_jac, jacres.rf_jac, jacres.rh_jac]
  jacobians_stacked = casadi.vertcat(*jacobians)
  tau_total = tau + casadi.transpose(jacobians_stacked) @ f
  cpin.aba(cmodel, cdata, q, v, tau_total)
  return cdata.ddq


def main(args: argparse.Namespace):
  logger = get_logger("trajopt", stdout_level=args.log_level)

  logger.info("Loading pinocchio model for go2")
  joint_model = pin.JointModelComposite(2)
  joint_model.addJoint(pin.JointModelTranslation())
  joint_model.addJoint(pin.JointModelSphericalZYX())
  desc = load_robot_description("go2_description", joint_model)
  model, data = desc.model, desc.data

  frame_names = [frame.name for frame in model.frames]
  logger.debug(f"Model[nq={model.nq}, nv={model.nv}] | frames: {frame_names}")

  logger.info("Setting up symbolic pinocchio model and data")
  cmodel = cpin.Model(model)
  cdata = cmodel.createData()

  q_rand = np.random.randn(model.nq)
  v_rand = np.random.randn(model.nv)
  tau_rand = np.random.randn(model.nv)
  f_rand = np.random.randn(12)

  logger.info("Creating symbolic variables")
  q_sym = casadi.SX.sym("q", model.nq)
  v_sym = casadi.SX.sym("v", model.nv)
  tau_sym = casadi.SX.sym("tau", model.nv)
  f_sym = casadi.SX.sym("f", 12)  # 3 force elements per foot, 4 feet

  logger.info("Creating casadi functions for forward kinematics")
  fk_res_casadi = forward_kinematics_casadi(cmodel, cdata, q_sym)
  fn_fk_com = casadi.Function("fk_com", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.com_pos])
  fn_fk_lf = casadi.Function("fk_lf", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lf_pos])
  fn_fk_lh = casadi.Function("fk_lh", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lh_pos])
  fn_fk_rf = casadi.Function("fk_rf", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rf_pos])
  fn_fk_rh = casadi.Function("fk_rh", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rh_pos])
  fn_fk_lf_knee = casadi.Function("fk_lf_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lf_knee_pos])
  fn_fk_lh_knee = casadi.Function("fk_lh_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lh_knee_pos])
  fn_fk_rf_knee = casadi.Function("fk_rf_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rf_knee_pos])
  fn_fk_rh_knee = casadi.Function("fk_rh_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rh_knee_pos])

  logger.info("Checking forward kinematics")
  assert np.allclose(forward_kinematics(model, data, q_rand).com_pos, np.array(fn_fk_com(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).lf_pos, np.array(fn_fk_lf(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).lh_pos, np.array(fn_fk_lh(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).rf_pos, np.array(fn_fk_rf(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).rh_pos, np.array(fn_fk_rh(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).lf_knee_pos, np.array(fn_fk_lf_knee(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).lh_knee_pos, np.array(fn_fk_lh_knee(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).rf_knee_pos, np.array(fn_fk_rf_knee(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))
  assert np.allclose(forward_kinematics(model, data, q_rand).rh_knee_pos, np.array(fn_fk_rh_knee(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))

  logger.info("Creating casadi functions for jacobians")
  jac_res_casadi = compute_jacobians_casadi(cmodel, cdata, q_sym)
  fn_jac_com = casadi.Function("jac_com", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.com_jac])
  fn_jac_lf = casadi.Function("jac_lf", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.lf_jac])
  fn_jac_lh = casadi.Function("jac_lh", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.lh_jac])
  fn_jac_rf = casadi.Function("jac_rf", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.rf_jac])
  fn_jac_rh = casadi.Function("jac_rh", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.rh_jac])

  logger.info("Checking jacobians")
  assert np.allclose(compute_jacobians(model, data, q_rand).com_jac, np.array(fn_jac_com(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1))
  assert np.allclose(compute_jacobians(model, data, q_rand).lf_jac, np.array(fn_jac_lf(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1))
  assert np.allclose(compute_jacobians(model, data, q_rand).lh_jac, np.array(fn_jac_lh(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1))
  assert np.allclose(compute_jacobians(model, data, q_rand).rf_jac, np.array(fn_jac_rf(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1))
  assert np.allclose(compute_jacobians(model, data, q_rand).rh_jac, np.array(fn_jac_rh(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1))

  logger.info("Creating casadi functions for jacobian derivatives")
  jac_dot_res_casadi = compute_jacobian_derivatives_casadi(cmodel, cdata, q_sym, v_sym)
  fn_jac_dot_com = casadi.Function("jac_dot_com", [q_sym, v_sym, tau_sym, f_sym], [jac_dot_res_casadi.com_jac_dot])
  fn_jac_dot_lf = casadi.Function("jac_dot_lf", [q_sym, v_sym, tau_sym, f_sym], [jac_dot_res_casadi.lf_jac_dot])
  fn_jac_dot_lh = casadi.Function("jac_dot_lh", [q_sym, v_sym, tau_sym, f_sym], [jac_dot_res_casadi.lh_jac_dot])
  fn_jac_dot_rf = casadi.Function("jac_dot_rf", [q_sym, v_sym, tau_sym, f_sym], [jac_dot_res_casadi.rf_jac_dot])
  fn_jac_dot_rh = casadi.Function("jac_dot_rh", [q_sym, v_sym, tau_sym, f_sym], [jac_dot_res_casadi.rh_jac_dot])

  logger.info("Checking jacobian derivatives")
  jac_dot_numerical = compute_jacobian_derivatives(model, data, q_rand, v_rand)
  assert np.allclose(jac_dot_numerical.com_jac_dot, np.array(fn_jac_dot_com(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1), atol=1e-6)
  assert np.allclose(jac_dot_numerical.lf_jac_dot, np.array(fn_jac_dot_lf(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1), atol=1e-6)
  assert np.allclose(jac_dot_numerical.lh_jac_dot, np.array(fn_jac_dot_lh(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1), atol=1e-6)
  assert np.allclose(jac_dot_numerical.rf_jac_dot, np.array(fn_jac_dot_rf(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1), atol=1e-6)
  assert np.allclose(jac_dot_numerical.rh_jac_dot, np.array(fn_jac_dot_rh(q_rand, v_rand, tau_rand, f_rand)).reshape(3, -1), atol=1e-6)
  logger.info("✓ Jacobian derivatives verification passed!")

  logger.info("Creating casadi functions for forward dynamics")
  fn_fd = casadi.Function("fd", [q_sym, v_sym, tau_sym, f_sym], [forward_dynamics_casadi(cmodel, cdata, q_sym, v_sym, tau_sym, f_sym)])

  logger.info("Checking forward dynamics")
  assert np.allclose(forward_dynamics(model, data, q_rand, v_rand, tau_rand, f_rand), np.array(fn_fd(q_rand, v_rand, tau_rand, f_rand)).reshape(-1))

  logger.info("Creating optimization problem")
  opti = casadi.Opti()

  q_opt = opti.variable(model.nq, args.num_steps + 1)
  v_opt = opti.variable(model.nv, args.num_steps + 1)
  tau_opt = opti.variable(model.nv, args.num_steps + 1)
  f_opt = opti.variable(12, args.num_steps + 1)
  logger.debug(f"optimization variable shapes: q_opt: {q_opt.shape}, v_opt: {v_opt.shape}, tau_opt: {tau_opt.shape}, f_opt: {f_opt.shape}")

  logger.info("Adding dynamics constraints")
  for t in range(args.num_steps):
    dt = args.dt
    opti.subject_to(q_opt[:, t + 1] == q_opt[:, t] + v_opt[:, t] * dt) # integrate position
    opti.subject_to(v_opt[:, t + 1] == v_opt[:, t] + fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) * dt) # integrate velocity

  logger.info("Adding initial position constraints")
  q_initial = np.array([0.0,  0,  0.33, 0,0,0, 0,0.806, -1.802, 0,0.806, -1.802, 0,0.806, -1.802, 0,0.806, -1.802])
  opti.subject_to(q_opt[:, 0] == q_initial)

  logger.info("Adding initial velocity constraints")
  v_initial = np.zeros(model.nv)
  opti.subject_to(v_opt[:, 0] == v_initial)

  logger.info("Adding final position constraints")
  q_final = np.array([1.5,  0,  0.33, 0.0,0, 0, 0,0.806, -1.802, 0,0.806, -1.802, 0,0.806, -1.802, 0,0.806, -1.802])
  opti.subject_to(q_opt[:, -1] == q_final)

  logger.info("Adding final velocity constraints")
  v_final = np.zeros(model.nv)
  opti.subject_to(v_opt[:, -1] == v_final)

  logger.info("Adding not base actuation constraints")
  for t in range(args.num_steps):
    opti.subject_to(tau_opt[:6, t] == 0)

  logger.info("Adding knee height constraints")
  knee_clearance = args.knee_clearance  # minimum clearance from ground

  mu = 0.8
  logger.info(f"Adding stance 1 friction cone constraints and contact constraints: mu = {mu}")
  stance1_start = int(0)
  stance1_end = int(0.3 * args.num_steps)
  
  # Store initial foot positions for contact constraints
  fk_initial = forward_kinematics(model, data, q_initial)
  initial_foot_positions = [fk_initial.lf_pos, fk_initial.lh_pos, fk_initial.rf_pos, fk_initial.rh_pos]
  
  for t in range(stance1_start, stance1_end):
    for i in range(4):
      fx = f_opt[3*i, t]
      fy = f_opt[3*i+1, t]
      fz = f_opt[3*i+2, t]
      opti.subject_to(fz >= 0)
      opti.subject_to(fx <= mu * fz)
      opti.subject_to(fx >= -mu * fz)
      opti.subject_to(fy <= mu * fz)
      opti.subject_to(fy >= -mu * fz)

    # Foot position constraints (feet stay at initial positions)
    opti.subject_to(fn_fk_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_foot_positions[0])
    opti.subject_to(fn_fk_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_foot_positions[1])
    opti.subject_to(fn_fk_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_foot_positions[2])
    opti.subject_to(fn_fk_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_foot_positions[3])
    
    # Contact constraints: -J_s^T * qdd = Jd_s * qd
    # This ensures that foot accelerations are zero (contact constraint from the paper)
    if t < stance1_end - 1:  # Don't apply on last timestep to avoid derivative issues
      qdd = fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      
      # For each foot, apply the contact constraint: J_s * qdd + Jd_s * qd = 0
      jac_lf = fn_jac_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_lh = fn_jac_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_rf = fn_jac_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_rh = fn_jac_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      
      jac_dot_lf = fn_jac_dot_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_dot_lh = fn_jac_dot_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_dot_rf = fn_jac_dot_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_dot_rh = fn_jac_dot_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      
      # Contact constraint for each foot: J_s * qdd + Jd_s * qd = 0 (foot acceleration = 0)
      opti.subject_to(jac_lf @ qdd + jac_dot_lf @ v_opt[:, t] == 0)
      opti.subject_to(jac_lh @ qdd + jac_dot_lh @ v_opt[:, t] == 0)
      opti.subject_to(jac_rf @ qdd + jac_dot_rf @ v_opt[:, t] == 0)
      opti.subject_to(jac_rh @ qdd + jac_dot_rh @ v_opt[:, t] == 0)

  logger.info("Adding flight friction cone constraints")
  flight_start = int(stance1_end)
  flight_end = int(0.7 * args.num_steps)
  for t in range(flight_start, flight_end):
    for i in range(4):
      fx = f_opt[3*i, t]
      fy = f_opt[3*i+1, t]
      fz = f_opt[3*i+2, t]
      opti.subject_to(fx == 0)
      opti.subject_to(fy == 0)
      opti.subject_to(fz == 0)

  logger.info(f"Adding stance 2 friction cone constraints and contact constraints: mu = {mu}")
  stance2_start = int(flight_end)
  stance2_end = int(args.num_steps) + 1
  
  # Store final foot positions for contact constraints
  fk_final = forward_kinematics(model, data, q_final)
  final_foot_positions = [fk_final.lf_pos, fk_final.lh_pos, fk_final.rf_pos, fk_final.rh_pos]
  
  for t in range(stance2_start, stance2_end):
    for i in range(4):
      if t == stance2_end - 1:
        continue
      fx = f_opt[3*i, t]
      fy = f_opt[3*i+1, t]
      fz = f_opt[3*i+2, t]
      opti.subject_to(fz >= 0)
      opti.subject_to(fx <= mu * fz)
      opti.subject_to(fx >= -mu * fz)
      opti.subject_to(fy <= mu * fz)
      opti.subject_to(fy >= -mu * fz)
    
    # Foot position constraints (feet stay at final positions)
    opti.subject_to(fn_fk_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_foot_positions[0])
    opti.subject_to(fn_fk_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_foot_positions[1])
    opti.subject_to(fn_fk_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_foot_positions[2])
    opti.subject_to(fn_fk_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_foot_positions[3])

    # Knee clearance constraints
    opti.subject_to(fn_fk_lf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > knee_clearance)
    opti.subject_to(fn_fk_lh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > knee_clearance)
    opti.subject_to(fn_fk_rf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > knee_clearance)
    opti.subject_to(fn_fk_rh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > knee_clearance)
    
    # Contact constraints: -J_s^T * qdd = Jd_s * qd
    # This ensures that foot accelerations are zero (contact constraint from the paper)
    if t < stance2_end - 1:  # Don't apply on last timestep to avoid derivative issues
      qdd = fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      
      # For each foot, apply the contact constraint: J_s * qdd + Jd_s * qd = 0
      jac_lf = fn_jac_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_lh = fn_jac_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_rf = fn_jac_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_rh = fn_jac_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      
      jac_dot_lf = fn_jac_dot_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_dot_lh = fn_jac_dot_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_dot_rf = fn_jac_dot_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      jac_dot_rh = fn_jac_dot_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])
      
      # Contact constraint for each foot: J_s * qdd + Jd_s * qd = 0 (foot acceleration = 0)
      opti.subject_to(jac_lf @ qdd + jac_dot_lf @ v_opt[:, t] == 0)
      opti.subject_to(jac_lh @ qdd + jac_dot_lh @ v_opt[:, t] == 0)
      opti.subject_to(jac_rf @ qdd + jac_dot_rf @ v_opt[:, t] == 0)
      opti.subject_to(jac_rh @ qdd + jac_dot_rh @ v_opt[:, t] == 0)

  ## Cost function
  logger.info("Adding cost function")
  total_cost = 0.0
  for t in range(args.num_steps):
    total_cost += casadi.sumsqr(q_opt[6:, t] - q_initial[6:]) * 30
    total_cost += casadi.sumsqr(v_opt[:, t]) * 10
    total_cost += casadi.sumsqr(tau_opt[:, t]) * 1
    total_cost += casadi.sumsqr(f_opt[:, t]) * 0.1
  opti.minimize(total_cost)

  logger.info("Solving optimization problem")
  opti.solver("ipopt", {"expand": False}, {"max_iter": 10000})
  try:
    sol = opti.solve()
  except Exception as e:
    logger.error(f"Error solving optimization problem: {e}")
    return

  logger.info(f"Optimization problem solved in {sol.stats()['iter_count']} iterations")
  logger.info("Solved optimization problem")

  if args.visualize:
    t_start = 0.0
    logger.info("Creating robot logger")
    robot_logger = RobotLogger.from_zoo("go2_description")
    logger.info("Visualizing optimization problem")
    rr.init("dsa", spawn=True)

    logger.info("Logging robot states")
    for i in range(args.num_steps + 1):
      q_i = sol.value(q_opt)[:,i]
      base_position = q_i[:3]
      base_orientation = R.from_euler("zyx", q_i[3:6], degrees=False).as_quat(scalar_first=False)

      joint_positions = {
          "FL_hip_joint": q_i[6], 
          "FL_thigh_joint": q_i[7],
          "FL_calf_joint": q_i[8],
          "FR_hip_joint": q_i[9],
          "FR_thigh_joint": q_i[10],
          "FR_calf_joint": q_i[11],
          "RL_hip_joint": q_i[12],
          "RL_thigh_joint": q_i[13],
          "RL_calf_joint": q_i[14],
          "RR_hip_joint": q_i[15],
          "RR_thigh_joint": q_i[16],
          "RR_calf_joint": q_i[17],
      }

      robot_logger.log_state(
          logtime=t_start + i * args.dt,
          base_position=base_position,
          base_orientation=base_orientation,
          joint_positions=joint_positions
      )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--log_level", type=str, default="info")
  parser.add_argument("--visualize", action="store_true")
  parser.add_argument("--num_steps", type=int, default=30)
  parser.add_argument("--dt", type=float, default=0.05)
  parser.add_argument("--knee_clearance", type=float, default=0.08, help="Minimum knee height above ground (meters)")
  args = parser.parse_args()

  if args.visualize:
    rerun_initialize("trajopt")

  main(args)