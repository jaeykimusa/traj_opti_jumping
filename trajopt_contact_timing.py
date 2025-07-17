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
                 num_steps: int = 30,
                 dt: float = 0.02,
                 knee_clearance: float = 0.08,
                 robot_description: str = "go2_description",
                 friction_coefficient: float = 0.8,
                 stance1_end_fraction: float = 0.3,
                 flight_end_fraction: float = 0.7,
                 joint_position_weight: float = 30.0,
                 velocity_weight: float = 10.0,
                 torque_weight: float = 1.0,
                 force_weight: float = 0.1,
                 max_iterations: int = 10000,
                 contact_sequence: List[List[int]] = None):
        
        # Configuration parameters
        self.log_level = log_level
        self.visualize = visualize
        self.num_steps = num_steps
        self.dt = dt
        self.knee_clearance = knee_clearance
        self.robot_description = robot_description
        self.friction_coefficient = friction_coefficient
        self.stance1_end_fraction = stance1_end_fraction
        self.flight_end_fraction = flight_end_fraction
        self.joint_position_weight = joint_position_weight
        self.velocity_weight = velocity_weight
        self.torque_weight = torque_weight
        self.force_weight = force_weight
        self.max_iterations = max_iterations
        self.dt_c = 0.025  # contact phase dt
        self.dt_f = 0.1 # Flight phase dt
        self.contact_sequence = contact_sequence 
        
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

        # Decision variables
        q_opt = opti.variable(self.model.nq, self.num_steps + 1)
        v_opt = opti.variable(self.model.nv, self.num_steps + 1)
        tau_opt = opti.variable(self.model.nv, self.num_steps + 1)
        f_opt = opti.variable(12, self.num_steps + 1)
        
        self.logger.debug(f"optimization variable shapes: q_opt: {q_opt.shape}, v_opt: {v_opt.shape}, tau_opt: {tau_opt.shape}, f_opt: {f_opt.shape}")

        # Initial and final configurations
        q_initial = np.array([0.0, 0, 0.33, 0, 0, 0, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802])
        v_initial = np.zeros(self.model.nv)
        q_final = np.array([1.5, 0, 0.33, 0, 0, 0, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802, 0, 0.806, -1.802])
        v_final = np.zeros(self.model.nv)

        # Dynamics constraints
        self.logger.info("Adding dynamics constraints")
        for t in range(self.num_steps):
            dt = self.dt
            opti.subject_to(q_opt[:, t + 1] == q_opt[:, t] + v_opt[:, t] * dt)  # integrate position
            opti.subject_to(v_opt[:, t + 1] == v_opt[:, t] + self.fn_fd(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) * dt)  # integrate velocity

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

        # Phase definitions
        stance1_start = int(0)
        stance1_end = int(self.stance1_end_fraction * self.num_steps)
        flight_start = int(stance1_end)
        flight_end = int(self.flight_end_fraction * self.num_steps)
        stance2_start = int(flight_end)
        stance2_end = int(self.num_steps) + 1

        # Stance 1 constraints
        self.logger.info(f"Adding stance 1 friction cone constraints: mu = {self.friction_coefficient}")
        for t in range(stance1_start, stance1_end):
            for i in range(4):
                fx = f_opt[3*i, t]
                fy = f_opt[3*i+1, t]
                fz = f_opt[3*i+2, t]
                opti.subject_to(fz >= 0)
                opti.subject_to(fx <= self.friction_coefficient * fz)
                opti.subject_to(fx >= -self.friction_coefficient * fz)
                opti.subject_to(fy <= self.friction_coefficient * fz)
                opti.subject_to(fy >= -self.friction_coefficient * fz)

            # Fixed foot positions during stance
            initial_fk = self.forward_kinematics(self.model, self.data, q_initial)
            opti.subject_to(self.fn_fk_lf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.lf_pos)
            opti.subject_to(self.fn_fk_lh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.lh_pos)
            opti.subject_to(self.fn_fk_rf(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.rf_pos)
            opti.subject_to(self.fn_fk_rh(q_opt[:, t], v_opt[:, t], tau_opt[:, t], f_opt[:, t]) == initial_fk.rh_pos)

        # Flight phase constraints
        self.logger.info("Adding flight phase constraints")
        for t in range(flight_start, flight_end):
            for i in range(4):
                fx = f_opt[3*i, t]
                fy = f_opt[3*i+1, t]
                fz = f_opt[3*i+2, t]
                opti.subject_to(fx == 0)
                opti.subject_to(fy == 0)
                opti.subject_to(fz == 0)

        # Stance 2 constraints
        self.logger.info(f"Adding stance 2 friction cone constraints: mu = {self.friction_coefficient}")
        for t in range(stance2_start, stance2_end):
            if t == stance2_end - 1:
                continue
                
            for i in range(4):
                fx = f_opt[3*i, t]
                fy = f_opt[3*i+1, t]
                fz = f_opt[3*i+2, t]
                opti.subject_to(fz >= 0)
                opti.subject_to(fx <= self.friction_coefficient * fz)
                opti.subject_to(fx >= -self.friction_coefficient * fz)
                opti.subject_to(fy <= self.friction_coefficient * fz)
                opti.subject_to(fy >= -self.friction_coefficient * fz)
            
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

        # Cost function
        self.logger.info("Adding cost function")
        total_cost = 0.0
        for t in range(self.num_steps):
            total_cost += casadi.sumsqr(q_opt[6:, t] - q_initial[6:]) * self.joint_position_weight
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

        t_start = 0.0
        self.logger.info("Creating robot logger")
        robot_logger = RobotLogger.from_zoo(self.robot_description)
        self.logger.info("Visualizing optimization solution")
        rr.init("trajectory_optimization", spawn=True)

        self.logger.info("Logging robot states")
        q_traj = self.solution['q']
        
        for i in range(self.num_steps + 1):
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

            robot_logger.log_state(
                logtime=t_start + i * self.dt,
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


if __name__ == "__main__":

    contact_sequence = [
        [1, 1, 1, 1],  # stance
        [0, 0, 0, 0],  # flight
        [1, 1, 1, 1],  # landing
    ]

    trajopt = TrajectoryOptimization(
        visualize=True,
        num_steps=30,
        dt=0.02,
        knee_clearance=0.08,
        contact_sequence=contact_sequence
    )
    
    if trajopt.solve():
        trajopt.visualize_solution()
        print("Trajectory optimization completed successfully!")
    else:
        print("Trajectory optimization failed!")