#!/usr/bin/env python3

from trajopt_logging import get_logger
from mpac_logging.rerun.utils import rerun_initialize
from mpac_logging.rerun.robot_logger import RobotLogger

from robot_descriptions.loaders.pinocchio import load_robot_description

import pinocchio as pin
import pinocchio.casadi as cpin
import casadi

from dataclasses import dataclass
from typing import Union, List, Optional

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt



class TrajectoryOptimization:
    
    
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


    def __init__(self, 
                 log_level: str = "info",
                 visualize: bool = False,
                #  dt: float = 0.02,
                 knee_clearance: float = 0.01,
                 robot_description: str = "go2_description",
                 mu: float = 0.4,
                 stance1_end_fraction: float = 0.25,
                 takeoff_end_fraction: float = 0.35,
                 flight_end_fraction: float = 0.75,
                 joint_position_weight: float = 1.5,
                 velocity_weight: float = 1.0,
                 torque_weight: float = 0.01,
                 force_weight: float = 0.001,
                 max_iterations: int = 10000,
                 contact_sequence: List[List[int]] = None):
        
        # Configuration parameters
        self.log_level = log_level
        self.visualize = visualize
        # self.dt = dt
        self.knee_clearance = knee_clearance
        self.robot_description = robot_description
        self.mu = mu
        self.stance1_end_fraction = stance1_end_fraction
        self.flight_end_fraction = flight_end_fraction
        self.joint_position_weight = joint_position_weight
        self.velocity_weight = velocity_weight
        self.torque_weight = torque_weight
        self.force_weight = force_weight
        self.max_iterations = max_iterations
        self.dt_c = 0.04 #self.T_jump / ((stance1_end_fraction + 4*(flight_end_fraction-stance1_end_fraction) + (1-flight_end_fraction)) * num_steps) #0.025  # contact phase dt
        self.dt_f = 0.04 # * self.dt_c # Flight phase dt

        self.T_stance = 0.60
        self.stance_steps = int(self.T_stance / self.dt_c)  # Number of steps in stance phase

        self.T_take_off = 0.40
        self.take_off_steps = int(self.T_take_off / self.dt_c)  # Number of steps in take-off phase

        self.T_flight = .72 #0.68
        self.flight_steps = int(self.T_flight / self.dt_f)  # Number of steps in flight phase

        self.T_landing = 0.32
        self.landing_steps = int(self.T_landing / self.dt_c)  # Number of steps in landing phase


        self.num_steps = self.stance_steps + self.take_off_steps + self.flight_steps + self.landing_steps  # Total number of steps
        self.T_jump = self.T_stance + self.T_take_off + self.T_flight + self.T_landing # 1.86 sec
    
        self.contact_sequence = contact_sequence

        # self.contact_sequence = contact_sequence 
        # print(self.dt_c)
        # print(self.dt_f)
        # exit()
        # Initialize logger and internal state
        self.logger = get_logger("trajopt", stdout_level=self.log_level)
        self.model = None
        self.data = None
        self.cmodel = None
        self.cdata = None
        self.solution = None
        
        # Casadi functions
        self.fn_fk_com = None
        self.fn_fk_lf = None
        self.fn_fk_lh = None
        self.fn_fk_rf = None
        self.fn_fk_rh = None
        self.fn_fk_lf_knee = None
        self.fn_fk_lh_knee = None
        self.fn_fk_rf_knee = None
        self.fn_fk_rh_knee = None
        self.fn_jac_com = None
        self.fn_jac_lf = None
        self.fn_jac_lh = None
        self.fn_jac_rf = None
        self.fn_jac_rh = None
        self.fn_fd = None

    def forward_kinematics(self, model, data, q: np.ndarray) -> 'TrajectoryOptimization.ForwardKinematicsResult':
        pin.framesForwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return self.ForwardKinematicsResult(
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

    def compute_jacobians(self, model, data, q: np.ndarray) -> 'TrajectoryOptimization.JacobianResult':
        pin.computeJointJacobians(model, data, q)
        pin.framesForwardKinematics(model, data, q)
        return self.JacobianResult(
            com_jac=pin.getFrameJacobian(model, data, model.getFrameId("base"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            lf_jac=pin.getFrameJacobian(model, data, model.getFrameId("FL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            lh_jac=pin.getFrameJacobian(model, data, model.getFrameId("RL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            rf_jac=pin.getFrameJacobian(model, data, model.getFrameId("FR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            rh_jac=pin.getFrameJacobian(model, data, model.getFrameId("RR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
        )

    def forward_dynamics(self, model, data, q: np.ndarray, v: np.ndarray, tau: np.ndarray, f: np.ndarray) -> np.ndarray:
        jacres = self.compute_jacobians(model, data, q)
        jacobians: List[np.ndarray] = [jacres.lf_jac, jacres.lh_jac, jacres.rf_jac, jacres.rh_jac]
        jacobians_stacked = np.concatenate(jacobians, axis=0)
        tau_total = tau + np.transpose(jacobians_stacked) @ f
        pin.aba(model, data, q, v, tau_total)
        return data.ddq

    def forward_kinematics_casadi(self, cmodel, cdata, q: casadi.SX) -> 'TrajectoryOptimization.ForwardKinematicsResult':
        cpin.framesForwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)
        return self.ForwardKinematicsResult(
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

    def compute_jacobians_casadi(self, cmodel, cdata, q: casadi.SX) -> 'TrajectoryOptimization.JacobianResult':
        cpin.computeJointJacobians(cmodel, cdata, q)
        cpin.framesForwardKinematics(cmodel, cdata, q)
        return self.JacobianResult(
            com_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("base"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            lf_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("FL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            lh_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("RL_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            rf_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("FR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
            rh_jac=cpin.getFrameJacobian(cmodel, cdata, cmodel.getFrameId("RR_foot"), pin.LOCAL_WORLD_ALIGNED)[:3, :],
        )

    def forward_dynamics_casadi(self, cmodel, cdata, q: casadi.SX, v: casadi.SX, tau: casadi.SX, f: casadi.SX) -> casadi.SX:
        jacres = self.compute_jacobians_casadi(cmodel, cdata, q)
        jacobians: List[casadi.SX] = [jacres.lf_jac, jacres.lh_jac, jacres.rf_jac, jacres.rh_jac]
        jacobians_stacked = casadi.vertcat(*jacobians)
        tau_total = tau + casadi.transpose(jacobians_stacked) @ f
        cpin.aba(cmodel, cdata, q, v, tau_total)
        return cdata.ddq

    def load_robot_model(self):
        """Load the robot model and setup symbolic versions"""
        self.logger.info(f"Loading pinocchio model for {self.robot_description}")
        joint_model = pin.JointModelComposite(2)
        joint_model.addJoint(pin.JointModelTranslation())
        joint_model.addJoint(pin.JointModelSphericalZYX())
        desc = load_robot_description(self.robot_description, joint_model)
        self.model, self.data = desc.model, desc.data

        frame_names = [frame.name for frame in self.model.frames]
        self.logger.debug(f"Model[nq={self.model.nq}, nv={self.model.nv}] | frames: {frame_names}")

        self.logger.info("Setting up symbolic pinocchio model and data")
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

    def setup_casadi_functions(self):
        """Setup all CasADi symbolic functions"""
        self.logger.info("Creating symbolic variables")
        q_sym = casadi.SX.sym("q", self.model.nq)
        v_sym = casadi.SX.sym("v", self.model.nv)
        tau_sym = casadi.SX.sym("tau", self.model.nv)
        f_sym = casadi.SX.sym("f", 12)  # 3 force elements per foot, 4 feet

        self.logger.info("Creating casadi functions for forward kinematics")
        fk_res_casadi = self.forward_kinematics_casadi(self.cmodel, self.cdata, q_sym)
        self.fn_fk_com = casadi.Function("fk_com", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.com_pos])
        self.fn_fk_lf = casadi.Function("fk_lf", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lf_pos])
        self.fn_fk_lh = casadi.Function("fk_lh", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lh_pos])
        self.fn_fk_rf = casadi.Function("fk_rf", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rf_pos])
        self.fn_fk_rh = casadi.Function("fk_rh", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rh_pos])
        self.fn_fk_lf_knee = casadi.Function("fk_lf_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lf_knee_pos])
        self.fn_fk_lh_knee = casadi.Function("fk_lh_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.lh_knee_pos])
        self.fn_fk_rf_knee = casadi.Function("fk_rf_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rf_knee_pos])
        self.fn_fk_rh_knee = casadi.Function("fk_rh_knee", [q_sym, v_sym, tau_sym, f_sym], [fk_res_casadi.rh_knee_pos])

        self.logger.info("Creating casadi functions for jacobians")
        jac_res_casadi = self.compute_jacobians_casadi(self.cmodel, self.cdata, q_sym)
        self.fn_jac_com = casadi.Function("jac_com", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.com_jac])
        self.fn_jac_lf = casadi.Function("jac_lf", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.lf_jac])
        self.fn_jac_lh = casadi.Function("jac_lh", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.lh_jac])
        self.fn_jac_rf = casadi.Function("jac_rf", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.rf_jac])
        self.fn_jac_rh = casadi.Function("jac_rh", [q_sym, v_sym, tau_sym, f_sym], [jac_res_casadi.rh_jac])

        self.logger.info("Creating casadi functions for forward dynamics")
        self.fn_fd = casadi.Function("fd", [q_sym, v_sym, tau_sym, f_sym], 
                                   [self.forward_dynamics_casadi(self.cmodel, self.cdata, q_sym, v_sym, tau_sym, f_sym)])


    def setup_optimization_problem(self):
        """Setup the trajectory optimization problem"""
        self.logger.info("Creating optimization problem")
        opti = casadi.Opti()

        # num_phases = len(self.contact_sequence)

        # Decision variables
        q_opt = opti.variable(self.model.nq, self.num_steps)
        v_opt = opti.variable(self.model.nv, self.num_steps)
        tau_opt = opti.variable(self.model.nv, self.num_steps)
        f_opt = opti.variable(12, self.num_steps)
        # t_opt = opti.variable()
        
        self.logger.debug(f"optimization variable shapes: q_opt: {q_opt.shape}, v_opt: {v_opt.shape}, tau_opt: {tau_opt.shape}, f_opt: {f_opt.shape}")

        # Initial and final configurations
        q_initial = np.array([1.5, 0, 0.33, 0, 0, 0, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802])
        v_initial = np.zeros(self.model.nv)
        q_final = np.array([0, 0, 0.33, 0, 2*np.pi, 0, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802])
        v_final = np.zeros(self.model.nv)

        # Boundary conditions
        self.logger.info("Adding boundary conditions")
        opti.subject_to(q_opt[:, 0] == q_initial)
        opti.subject_to(v_opt[:, 0] == v_initial)
        opti.subject_to(q_opt[:, -1] == q_final)
        opti.subject_to(v_opt[:, -1] == v_final)

        # No base actuation
        self.logger.info("Adding no base actuation constraints")
        for t in range(self.num_steps):
            opti.subject_to(tau_opt[:6, t] == 0)
        
        # tau_ub = 100
        # tau_lb = -100
        # joint_ub = np.array([1, 1.5, 0.5] * 4)
        # joint_lb = np.array([-1, -1.5, -3.0] * 4)

        # for t in range(self.num_steps):
        #     opti.subject_to(q_opt[6:, t] <= joint_ub)
        #     opti.subject_to(q_opt[6:, t] >= joint_lb)
        #     opti.subject_to(tau_opt[6:,t] <= tau_ub)
        #     opti.subject_to(tau_opt[6:,t] >= tau_lb)

        # Stance 1 constraints
        self.logger.info(f"Adding stance phase constraints.")
        for t in range(0, self.stance_steps):
            for i in range(4):
                fx = f_opt[3*i, t]
                fy = f_opt[3*i+1, t]
                fz = f_opt[3*i+2, t]
                opti.subject_to(fz >= 0)
                opti.subject_to(fx <= self.mu * fz)
                opti.subject_to(fx >= -self.mu * fz)
                opti.subject_to(fy <= self.mu * fz)
                opti.subject_to(fy >= -self.mu * fz)

            # Fixed foot positions during stance
            initial_fk = self.forward_kinematics(self.model, self.data, q_initial)
            opti.subject_to(self.fn_fk_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.lf_pos)
            opti.subject_to(self.fn_fk_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.lh_pos)
            opti.subject_to(self.fn_fk_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.rf_pos)
            opti.subject_to(self.fn_fk_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.rh_pos)

            opti.subject_to(q_opt[:, t + 1] == q_opt[:, t] + v_opt[:, t] * self.dt_c)  # integrate position
            opti.subject_to(v_opt[:, t + 1] == v_opt[:, t] + self.fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) * self.dt_c)  # integrate velocity


        self.logger.info(f"Adding take-off phase constraints. ")
        for t in range(self.stance_steps, self.stance_steps+self.take_off_steps):
            for i in range(0,4):
                if i % 2 == 0:  # front foot
                    fx = f_opt[3*i, t]
                    fy = f_opt[3*i+1, t]
                    fz = f_opt[3*i+2, t]
                    opti.subject_to(fx == 0)
                    opti.subject_to(fy == 0)
                    opti.subject_to(fz == 0)
                else:
                    # rear foot
                    fx = f_opt[3*i, t]
                    fy = f_opt[3*i+1, t]
                    fz = f_opt[3*i+2, t]
                    opti.subject_to(fz >= 0)
                    opti.subject_to(fx <= self.mu * fz)
                    opti.subject_to(fx >= -self.mu * fz)
                    opti.subject_to(fy <= self.mu * fz)
                    opti.subject_to(fy >= -self.mu * fz)

            # Fixed foot positions during stance
            initial_fk = self.forward_kinematics(self.model, self.data, q_initial)
            opti.subject_to(self.fn_fk_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.lf_pos)
            opti.subject_to(self.fn_fk_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.lh_pos)
            opti.subject_to(self.fn_fk_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.rf_pos)
            opti.subject_to(self.fn_fk_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.rh_pos)

            # Knee clearance constraints
            opti.subject_to(self.fn_fk_lf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_lh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_rf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_rh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)

            opti.subject_to(q_opt[:, t + 1] == q_opt[:, t] + v_opt[:, t] * self.dt_c)  # integrate position
            opti.subject_to(v_opt[:, t + 1] == v_opt[:, t] + self.fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) * self.dt_c)  # integrate velocity


        # Flight phase constraints
        self.logger.info("Adding flight phase constraints.")
        for t in range(self.stance_steps+self.take_off_steps, self.stance_steps+self.take_off_steps+self.flight_steps):
            for i in range(4):
                fx = f_opt[3*i, t]
                fy = f_opt[3*i+1, t]
                fz = f_opt[3*i+2, t]
                opti.subject_to(fx == 0)
                opti.subject_to(fy == 0)
                opti.subject_to(fz == 0)

            # Knee clearance constraints
            opti.subject_to(self.fn_fk_lf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_lh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_rf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_rh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)

            opti.subject_to(q_opt[:, t + 1] == q_opt[:, t] + v_opt[:, t] * self.dt_f)  # integrate position
            opti.subject_to(v_opt[:, t + 1] == v_opt[:, t] + self.fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) * self.dt_f)  # integrate velocity

        # Landing phase constraints
        self.logger.info(f"Adding landing phase constraints.")
        for t in range(self.stance_steps+self.take_off_steps+self.flight_steps, self.num_steps):
            if t == self.num_steps - 1:
                continue
                
            for i in range(4):
                fx = f_opt[3*i, t]
                fy = f_opt[3*i+1, t]
                fz = f_opt[3*i+2, t]
                opti.subject_to(fz >= 0)
                opti.subject_to(fx <= self.mu * fz)
                opti.subject_to(fx >= -self.mu * fz)
                opti.subject_to(fy <= self.mu * fz)
                opti.subject_to(fy >= -self.mu * fz)
            
            # Fixed foot positions during stance
            final_fk = self.forward_kinematics(self.model, self.data, q_final)
            opti.subject_to(self.fn_fk_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_fk.lf_pos)
            opti.subject_to(self.fn_fk_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_fk.lh_pos)
            opti.subject_to(self.fn_fk_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_fk.rf_pos)
            opti.subject_to(self.fn_fk_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == final_fk.rh_pos)

            # Knee clearance constraints
            opti.subject_to(self.fn_fk_lf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_lh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_rf_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)
            opti.subject_to(self.fn_fk_rh_knee(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t])[2] > self.knee_clearance)

            opti.subject_to(q_opt[:, t + 1] == q_opt[:, t] + v_opt[:, t] * self.dt_c)  # integrate position
            opti.subject_to(v_opt[:, t + 1] == v_opt[:, t] + self.fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) * self.dt_c)  # integrate velocity

        # Cost function
        self.logger.info("Adding cost function")
        total_cost = 0.0
        for t in range(self.num_steps):
            total_cost += casadi.sumsqr(q_opt[:6, t]) * self.joint_position_weight
            total_cost += casadi.sumsqr(q_opt[6:, t]) * (self.joint_position_weight-0.5)
            total_cost += casadi.sumsqr(v_opt[:, t]) * self.velocity_weight
            total_cost += casadi.sumsqr(tau_opt[:, t]) * self.torque_weight
            total_cost += casadi.sumsqr(f_opt[:, t]) * self.force_weight
        opti.minimize(total_cost)

        return opti, q_opt, v_opt, tau_opt, f_opt

    def solve(self):
        """Solve the trajectory optimization problem"""
        # Load robot model and setup functions
        self.load_robot_model()
        self.setup_casadi_functions()
        
        # Setup and solve optimization
        opti, q_opt, v_opt, tau_opt, f_opt = self.setup_optimization_problem()
        
        self.logger.info("Solving optimization problem")
        opti.solver("ipopt", {"expand": False}, {"max_iter": self.max_iterations})
        
        try:
            sol = opti.solve()
            self.solution = {
                'q': sol.value(q_opt),
                'v': sol.value(v_opt),
                'tau': sol.value(tau_opt),
                'f': sol.value(f_opt),
                'stats': sol.stats()
            }
            self.logger.info(f"Optimization problem solved in {sol.stats()['iter_count']} iterations")
        except Exception as e:
            self.logger.error(f"Error solving optimization problem: {e}")
            return False
            
        return True

    def visualize_solution(self):
        """Visualize the solution using rerun"""
        if self.solution is None:
            self.logger.error("No solution to visualize. Run solve() first.")
            return
            
        if not self.visualize:
            self.logger.info("Visualization disabled")
            return

        time = 0.0
        self.logger.info("Creating robot logger")
        robot_logger = RobotLogger.from_zoo(self.robot_description)
        self.logger.info("Visualizing optimization solution")
        rr.init("trajectory_optimization", spawn=True)

        self.logger.info("Logging robot states")
        q_traj = self.solution['q']
        
        for i in range(self.num_steps):
            q_i = q_traj[:, i]
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
            if 0 <= i < self.stance_steps:
                time += self.dt_c
                robot_logger.log_state(
                    logtime=time,
                    base_position=base_position,
                    base_orientation=base_orientation,
                    joint_positions=joint_positions
                )
            elif self.stance_steps <= i < self.stance_steps+self.take_off_steps:
                time += self.dt_c
                robot_logger.log_state(
                    logtime=time,
                    base_position=base_position,
                    base_orientation=base_orientation,
                    joint_positions=joint_positions
                )
            elif self.stance_steps+self.take_off_steps <= i < self.stance_steps+self.take_off_steps+self.flight_steps:
                time += self.dt_f
                robot_logger.log_state(
                    logtime=time,
                    base_position=base_position,
                    base_orientation=base_orientation,
                    joint_positions=joint_positions
                )
            else:
                time += self.dt_c
                robot_logger.log_state(
                    logtime=time,
                    base_position=base_position,
                    base_orientation=base_orientation,
                    joint_positions=joint_positions
                )

    def get_solution(self):
        """Get the optimization solution"""
        return self.solution

    def set_initial_configuration(self, q_initial: np.ndarray, v_initial: Optional[np.ndarray] = None):
        """Set custom initial configuration"""
        self.q_initial = q_initial
        self.v_initial = v_initial if v_initial is not None else np.zeros(len(q_initial) - 1)

    def set_final_configuration(self, q_final: np.ndarray, v_final: Optional[np.ndarray] = None):
        """Set custom final configuration"""
        self.q_final = q_final
        self.v_final = v_final if v_final is not None else np.zeros(len(q_final) - 1)


    def plots(self):
        fig, axs = plt.subplots(4, 3, figsize=(15, 8))  # Wider layout
        axs = axs.flatten()
        
        q = self.solution['q']
        v = self.solution['v']
        tau = self.solution['tau']
        f = self.solution['f']

        com_x = q[0,:]
        com_z = q[2,:]
        axs[0].plot(com_x, com_z, label="com")
        axs[0].set_title("com position ")
        axs[0].legend()
        axs[0].set_xlabel("x (m)")
        axs[0].set_ylabel("z (m)")

        axs[1].plot(q[3,:], label="phi")
        axs[1].set_title("pitch angle")
        axs[1].legend()
        axs[1].set_xlabel("N")
        axs[1].set_ylabel("Angle (rad)")
        
        axs[2].plot(v[0,:], label="v_x")
        axs[2].set_title("v_x")
        axs[2].legend()
        axs[2].set_xlabel("N")
        axs[2].set_ylabel("v_x (m/s)")

        axs[3].plot(v[2,:], label="v_z")
        axs[3].set_title("v_z")
        axs[3].legend()
        axs[3].set_xlabel("N")
        axs[3].set_ylabel("v_z (m/s)")

        axs[4].plot(q[7,:], label="thigh")
        axs[4].plot(q[8,:], label="calf")
        axs[4].set_title("Joint angle tracking for front leg")
        axs[4].legend()
        axs[4].set_xlabel("N")
        axs[4].set_ylabel("Angle (rad)")

        axs[5].plot(q[10,:], label="thigh")
        axs[5].plot(q[11,:], label="calf")
        axs[5].set_title("Joint angle tracking for rear leg")
        axs[5].legend()
        axs[5].set_xlabel("N")
        axs[5].set_ylabel("Angle (rad)")

        axs[6].plot(f[0,:], label="F_x")
        axs[6].plot(f[2,:], label="F_z")
        axs[6].set_title("Ground reaction force on front leg")
        axs[6].legend()
        axs[6].set_xlabel("N")
        axs[6].set_ylabel("F (N)")

        axs[7].plot(f[3,:], label="F_x")
        axs[7].plot(f[5,:], label="F_z")
        axs[7].set_title("Ground reaction force on rear leg")
        axs[7].legend()
        axs[7].set_xlabel("N")
        axs[7].set_ylabel("F (N)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    contact_sequence = [
        [1, 1, 1, 1],  # stance
        [0, 0, 1, 1],  # take off
        [0, 0, 0, 0],  # flight
        [1, 1, 1, 1],  # landing
    ]

    trajopt = TrajectoryOptimization(
        visualize=True,
        # num_steps=30,
        knee_clearance=0.08
        # contact_sequence=contact_sequence
    )
    
    if trajopt.solve():
        trajopt.visualize_solution()
        print("Trajectory optimization completed successfully!")
    else:
        print("Trajectory optimization failed!")
    
    trajopt.plots()