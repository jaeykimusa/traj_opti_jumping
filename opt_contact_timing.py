"""
Contact-timing and Trajectory Optimization for 3D Jumping on Quadruped Robots
Paper implementation by Chuong Nguyen and Quan Nguyen

This module generates optimal contact timings and reference trajectories
for full-body trajectory optimization.
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

@dataclass
class RobotParams:
    """Robot parameters for A1 quadruped"""
    mass: float = 12.0  # kg
    gravity: float = 9.81  # m/s^2
    inertia: np.ndarray = None  # Body inertia tensor
    mu: float = 0.6  # Friction coefficient
    f_min: float = 0.0  # Min normal force
    f_max: float = 150.0  # Max normal force per foot
    
    def __post_init__(self):
        if self.inertia is None:
            # Default inertia for A1 robot
            self.inertia = np.diag([0.5, 0.8, 0.6])

@dataclass
class JumpTask:
    """Define jumping task parameters"""
    p_initial: np.ndarray  # Initial CoM position
    p_final: np.ndarray    # Final CoM position
    R_initial: np.ndarray = None  # Initial rotation
    R_final: np.ndarray = None    # Final rotation
    
    def __post_init__(self):
        if self.R_initial is None:
            self.R_initial = np.eye(3)
        if self.R_final is None:
            self.R_final = np.eye(3)

class ContactTimingOptimizer:
    """
    Implements contact timing optimization using simplified rigid body dynamics
    as described in Section II-B of the paper.
    """
    
    def __init__(self, robot_params: RobotParams):
        self.robot = robot_params
        
        # Foot positions in body frame (A1 robot)
        self.r_foot_body = {
            'FR': np.array([0.183, -0.13, 0]),
            'FL': np.array([0.183, 0.13, 0]),
            'RR': np.array([-0.183, -0.13, 0]),
            'RL': np.array([-0.183, 0.13, 0])
        }
        
        # Optimization weights from paper
        self.weights = {
            'omega': 1e-2,    # Angular velocity
            'force': 1e-3,    # Ground reaction forces
            'rotation': 1e-1  # Rotation error
        }
        
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        
        R = ca.vertcat(
            ca.horzcat(1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)),
            ca.horzcat(2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)),
            ca.horzcat(2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2))
        )
        return R
    
    def optimize_contact_timing(self, 
                              contact_sequence: List[List[int]], 
                              jump_task: JumpTask,
                              T_bounds: Tuple[float, float] = (0.3, 2.0),
                              N_per_phase: int = 20) -> Dict:
        """
        Main optimization function that solves for optimal contact timings.
        
        Args:
            contact_sequence: List of contact patterns for each phase
                             e.g., [[1,1,1,1], [0,0,1,1], [0,0,0,0]]
            jump_task: JumpTask object defining initial and final states
            T_bounds: (T_min, T_max) bounds on total time
            N_per_phase: Number of discretization points per phase
            
        Returns:
            Dictionary containing optimal timings and reference trajectories
        """
        
        n_phases = len(contact_sequence)
        N_total = n_phases * N_per_phase
        
        # Create optimization problem
        opti = ca.Opti()
        
        # ============ Decision Variables ============
        
        # Phase durations (main optimization variables!)
        T_phases = []
        for i in range(n_phases):
            T_i = opti.variable()
            opti.subject_to(0.01 <= T_i)  # Min 10ms per phase
            opti.subject_to(T_i <= 1.0)   # Max 1s per phase
            T_phases.append(T_i)
        
        # Total time constraint
        T_total = sum(T_phases)
        opti.subject_to(T_bounds[0] <= T_total)
        opti.subject_to(T_total <= T_bounds[1])
        
        # State trajectory variables
        states = []
        for k in range(N_total + 1):
            state = {
                'p': opti.variable(3),        # Position
                'v': opti.variable(3),        # Velocity
                'q': opti.variable(4),        # Quaternion (for rotation)
                'omega': opti.variable(3)     # Angular velocity
            }
            states.append(state)
        
        # Control variables (GRF for each foot)
        controls = []
        for k in range(N_total):
            control = {
                'FR': opti.variable(3),
                'FL': opti.variable(3),
                'RR': opti.variable(3),
                'RL': opti.variable(3)
            }
            controls.append(control)
        
        # ============ Initial Conditions ============
        opti.subject_to(states[0]['p'] == jump_task.p_initial)
        opti.subject_to(states[0]['v'] == 0)
        opti.subject_to(states[0]['q'] == ca.DM([1, 0, 0, 0]))  # Identity quaternion
        opti.subject_to(states[0]['omega'] == 0)
        
        # ============ Final Conditions ============
        opti.subject_to(states[-1]['p'] == jump_task.p_final)
        # Final rotation handled through cost function
        
        # ============ Cost Function ============
        cost = 0
        
        # ============ Dynamics Constraints ============
        k_global = 0
        
        for phase_idx in range(n_phases):
            # Variable timestep for this phase
            dt = T_phases[phase_idx] / N_per_phase
            contacts = contact_sequence[phase_idx]
            
            for k_local in range(N_per_phase):
                k = k_global + k_local
                
                if k < N_total:
                    # Current state
                    p = states[k]['p']
                    v = states[k]['v']
                    q = states[k]['q']
                    omega = states[k]['omega']
                    
                    # Current rotation matrix
                    R = self.quaternion_to_rotation_matrix(q)
                    
                    # Total force and torque
                    F_total = ca.MX([0, 0, -self.robot.mass * self.robot.gravity])
                    tau_total = ca.MX.zeros(3, 1)
                    
                    # Process each foot
                    for foot_name, foot_pos_body in self.r_foot_body.items():
                        foot_idx = ['FR', 'FL', 'RR', 'RL'].index(foot_name)
                        
                        if contacts[foot_idx]:  # Foot in contact
                            f = controls[k][foot_name]
                            
                            # Add to total force
                            F_total += f
                            
                            # Foot position in world frame
                            r_foot = R @ foot_pos_body
                            
                            # Add to total torque
                            tau_total += ca.cross(r_foot, f)
                            
                            # Force constraints
                            opti.subject_to(0 <= f[2])
                            opti.subject_to(f[2] <= self.robot.f_max)
                            
                            # Friction cone
                            opti.subject_to(f[0]**2 + f[1]**2 <= (self.robot.mu * f[2])**2)
                            
                        else:  # No contact
                            opti.subject_to(controls[k][foot_name] == 0)
                    
                    # Translation dynamics: p_dot = v, v_dot = F/m
                    p_next = p + dt * v
                    v_next = v + dt * F_total / self.robot.mass
                    
                    # Rotation dynamics
                    # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, omega]
                    omega_quat = ca.vertcat(0, omega)
                    q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
                    q_next = q + dt * q_dot
                    
                    # Normalize quaternion
                    q_next = q_next / ca.norm_2(q_next)
                    
                    # Angular dynamics: I*omega_dot = tau - omega x (I*omega)
                    I_omega = self.robot.inertia @ omega
                    omega_cross_I_omega = ca.cross(omega, I_omega)
                    tau_body = R.T @ tau_total
                    omega_dot = ca.solve(self.robot.inertia, tau_body - omega_cross_I_omega)
                    omega_next = omega + dt * omega_dot
                    
                    # Apply dynamics constraints
                    opti.subject_to(states[k+1]['p'] == p_next)
                    opti.subject_to(states[k+1]['v'] == v_next)
                    opti.subject_to(states[k+1]['q'] == q_next)
                    opti.subject_to(states[k+1]['omega'] == omega_next)
                    
                    # Quaternion unit constraint
                    opti.subject_to(ca.sumsqr(states[k+1]['q']) == 1)
                    
                    # Add costs
                    cost += self.weights['omega'] * ca.dot(omega, omega)
                    for foot_name in ['FR', 'FL', 'RR', 'RL']:
                        f = controls[k][foot_name]
                        cost += self.weights['force'] * ca.dot(f, f)
                
            k_global += N_per_phase
        
        # Terminal cost for rotation
        q_final = states[-1]['q']
        R_final = self.quaternion_to_rotation_matrix(q_final)
        # Simple rotation error (could be improved)
        rotation_error = ca.trace(ca.DM.eye(3) - R_final.T @ jump_task.R_final)
        cost += self.weights['rotation'] * rotation_error
        
        # ============ Solve Optimization ============
        opti.minimize(cost)
        
        # Solver settings
        opts = {
            'ipopt.print_level': 3,
            'ipopt.max_iter': 500,
            'ipopt.tol': 1e-5,
            'ipopt.acceptable_tol': 1e-4
        }
        opti.solver('ipopt', opts)
        
        # Initial guess
        self._set_initial_guess(opti, T_phases, states, controls, 
                               contact_sequence, jump_task, N_per_phase)
        
        # Solve
        try:
            sol = opti.solve()
            
            # Extract results
            T_opt = [sol.value(T_phases[i]) for i in range(n_phases)]
            
            # Build time grid
            time_grid = []
            t = 0
            for i in range(n_phases):
                dt = T_opt[i] / N_per_phase
                for j in range(N_per_phase):
                    time_grid.append(t + j * dt)
                t += T_opt[i]
            time_grid.append(t)
            
            # Extract trajectories
            trajectory = {
                'time': np.array(time_grid),
                'position': np.array([sol.value(states[k]['p']) for k in range(N_total + 1)]),
                'velocity': np.array([sol.value(states[k]['v']) for k in range(N_total + 1)]),
                'quaternion': np.array([sol.value(states[k]['q']) for k in range(N_total + 1)]),
                'omega': np.array([sol.value(states[k]['omega']) for k in range(N_total + 1)]),
                'forces': {}
            }
            
            # Extract forces
            for foot_name in ['FR', 'FL', 'RR', 'RL']:
                trajectory['forces'][foot_name] = np.array([
                    sol.value(controls[k][foot_name]) for k in range(N_total)
                ])
            
            # Results summary
            results = {
                'success': True,
                'phase_durations': T_opt,
                'total_time': sum(T_opt),
                'trajectory': trajectory,
                'contact_sequence': contact_sequence,
                'cost': sol.value(cost)
            }
            
            # Print summary
            self._print_results_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def quaternion_multiply(self, q1, q2):
        """Quaternion multiplication q1 ⊗ q2"""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        
        return ca.vertcat(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )
    
    def _set_initial_guess(self, opti, T_phases, states, controls, 
                          contact_sequence, jump_task, N_per_phase):
        """Set initial guess for optimization"""
        
        # Initial guess for phase durations
        T_total_guess = 0.8
        n_phases = len(contact_sequence)
        for i in range(n_phases):
            opti.set_initial(T_phases[i], T_total_guess / n_phases)
        
        # Initial guess for states (linear interpolation)
        N_total = n_phases * N_per_phase
        for k in range(N_total + 1):
            alpha = k / N_total
            
            # Position: linear interpolation
            p_guess = (1 - alpha) * jump_task.p_initial + alpha * jump_task.p_final
            opti.set_initial(states[k]['p'], p_guess)
            
            # Velocity: constant
            v_guess = (jump_task.p_final - jump_task.p_initial) / T_total_guess
            opti.set_initial(states[k]['v'], v_guess)
            
            # Quaternion: identity
            opti.set_initial(states[k]['q'], [1, 0, 0, 0])
            
            # Angular velocity: zero
            opti.set_initial(states[k]['omega'], [0, 0, 0])
        
        # Initial guess for forces
        for k in range(N_total):
            phase_idx = k // N_per_phase
            contacts = contact_sequence[phase_idx]
            
            for i, foot_name in enumerate(['FR', 'FL', 'RR', 'RL']):
                if contacts[i]:
                    # Distribute weight evenly among contact feet
                    n_contacts = sum(contacts)
                    f_z = self.robot.mass * self.robot.gravity / n_contacts
                    opti.set_initial(controls[k][foot_name], [0, 0, f_z])
                else:
                    opti.set_initial(controls[k][foot_name], [0, 0, 0])
    
    def _print_results_summary(self, results):
        """Print optimization results summary"""
        print("\n" + "="*50)
        print("CONTACT TIMING OPTIMIZATION RESULTS")
        print("="*50)
        
        print(f"\nTotal time: {results['total_time']:.3f} s")
        
        # Calculate and print contact switching times
        print("\nContact Event Timings:")
        cumulative_time = 0.0
        in_flight = False
        
        for i, (duration, contacts) in enumerate(zip(results['phase_durations'], 
                                                     results['contact_sequence'])):
            if i == 0:
                print(f"  t = 0.000 s: START (Stance phase)")
            
            cumulative_time += duration
            
            # Detect contact changes
            if i < len(results['contact_sequence']) - 1:
                next_contacts = results['contact_sequence'][i + 1]
                
                # Check for takeoff (going from contact to no contact)
                if sum(contacts) > sum(next_contacts):
                    if sum(next_contacts) == 0:
                        print(f"  t = {cumulative_time:.3f} s: TAKEOFF (All feet leave ground)")
                        in_flight = True
                    else:
                        contact_str = ['FR', 'FL', 'RR', 'RL']
                        leaving_feet = [contact_str[j] for j in range(4) 
                                      if contacts[j] and not next_contacts[j]]
                        print(f"  t = {cumulative_time:.3f} s: Partial takeoff ({leaving_feet} leave ground)")
                
                # Check for landing (going from no/partial contact to more contact)
                elif sum(contacts) < sum(next_contacts):
                    if sum(contacts) == 0:
                        # Coming from flight
                        contact_str = ['FR', 'FL', 'RR', 'RL']
                        landing_feet = [contact_str[j] for j in range(4) if next_contacts[j]]
                        if sum(next_contacts) == 4:
                            print(f"  t = {cumulative_time:.3f} s: LANDING (All feet touch ground)")
                        else:
                            print(f"  t = {cumulative_time:.3f} s: Partial landing ({landing_feet} touch ground)")
                        in_flight = False
                    else:
                        # Additional feet making contact
                        contact_str = ['FR', 'FL', 'RR', 'RL']
                        new_feet = [contact_str[j] for j in range(4) 
                                   if not contacts[j] and next_contacts[j]]
                        print(f"  t = {cumulative_time:.3f} s: Additional landing ({new_feet} touch ground)")
        
        print(f"  t = {cumulative_time:.3f} s: END")
        
        print("\nPhase durations:")
        for i, (duration, contacts) in enumerate(zip(results['phase_durations'], 
                                                     results['contact_sequence'])):
            contact_str = ['FR', 'FL', 'RR', 'RL']
            active_feet = [contact_str[j] for j, c in enumerate(contacts) if c]
            if not active_feet:
                active_feet = ['FLIGHT']
            print(f"  Phase {i+1} ({', '.join(active_feet)}): {duration*1000:.1f} ms")
        
        # Print reference trajectory in 3xN format
        traj = results['trajectory']
        positions = traj['position'].T  # Transpose to get 3xN
        
        print("\nReference Trajectory (3 x N format):")
        print(f"  Shape: {positions.shape[0]} x {positions.shape[1]}")
        print("  First 5 points:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            print(f"    {axis}: [{', '.join(f'{p:.3f}' for p in positions[i, :5])} ...]")
        print("  Last 5 points:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            print(f"    {axis}: [... {', '.join(f'{p:.3f}' for p in positions[i, -5:])}]")
        
        # Trajectory statistics
        max_height = np.max(positions[2, :])
        max_velocity = np.max(np.linalg.norm(traj['velocity'], axis=1))
        
        print(f"\nTrajectory statistics:")
        print(f"  Max height: {max_height:.3f} m")
        print(f"  Max velocity: {max_velocity:.3f} m/s")
        print(f"  Jump distance: {positions[0, -1] - positions[0, 0]:.3f} m")
        print(f"  Lateral deviation: {abs(positions[1, -1] - positions[1, 0]):.3f} m")
        
        print(f"\nOptimization cost: {results['cost']:.6f}")
    
    def get_reference_trajectory_3xN(self, results):
        """
        Get reference trajectory in 3xN format
        
        Returns:
            positions: np.ndarray of shape (3, N) where rows are [x, y, z]
            times: np.ndarray of shape (N,) with time for each point
        """
        if not results['success']:
            return None, None
        
        positions = results['trajectory']['position'].T  # Transpose to 3xN
        times = results['trajectory']['time']
        
        return positions, times
    
    def get_contact_events(self, results):
        """
        Extract contact event timings
        
        Returns:
            dict with 'takeoff_time', 'landing_time', and other events
        """
        if not results['success']:
            return None
        
        events = {}
        cumulative_time = 0.0
        
        for i, (duration, contacts) in enumerate(zip(results['phase_durations'], 
                                                     results['contact_sequence'])):
            if i > 0:
                cumulative_time += results['phase_durations'][i-1]
            
            if i < len(results['contact_sequence']) - 1:
                next_contacts = results['contact_sequence'][i + 1]
                
                # Full takeoff (all feet leave ground)
                if sum(contacts) > 0 and sum(next_contacts) == 0:
                    events['takeoff_time'] = cumulative_time + duration
                
                # Landing (feet touch ground after flight)
                if sum(contacts) == 0 and sum(next_contacts) > 0:
                    events['landing_time'] = cumulative_time + duration
                
                # Partial transitions
                if sum(contacts) > sum(next_contacts) > 0:
                    events['partial_takeoff_time'] = cumulative_time + duration
        
        events['start_time'] = 0.0
        events['end_time'] = sum(results['phase_durations'])
        
        return events
    
    def plot_results(self, results):
        """Visualize optimization results"""
        if not results['success']:
            print("Cannot plot failed optimization")
            return
        
        traj = results['trajectory']
        time = traj['time']
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # Position
        ax = axes[0, 0]
        ax.plot(time, traj['position'][:, 0], 'r-', label='x')
        ax.plot(time, traj['position'][:, 1], 'g-', label='y')
        ax.plot(time, traj['position'][:, 2], 'b-', label='z')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True)
        
        # Velocity
        ax = axes[0, 1]
        ax.plot(time, traj['velocity'][:, 0], 'r-', label='vx')
        ax.plot(time, traj['velocity'][:, 1], 'g-', label='vy')
        ax.plot(time, traj['velocity'][:, 2], 'b-', label='vz')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.grid(True)
        
        # Angular velocity
        ax = axes[1, 0]
        ax.plot(time, traj['omega'][:, 0], 'r-', label='ωx')
        ax.plot(time, traj['omega'][:, 1], 'g-', label='ωy')
        ax.plot(time, traj['omega'][:, 2], 'b-', label='ωz')
        ax.set_ylabel('Angular velocity (rad/s)')
        ax.legend()
        ax.grid(True)
        
        # Total force
        ax = axes[1, 1]
        total_force = np.zeros((len(time)-1, 3))
        for foot_name in ['FR', 'FL', 'RR', 'RL']:
            total_force += traj['forces'][foot_name]
        ax.plot(time[:-1], total_force[:, 2], 'k-', linewidth=2)
        ax.axhline(y=self.robot.mass * self.robot.gravity, 
                  color='r', linestyle='--', label='Weight')
        ax.set_ylabel('Total vertical force (N)')
        ax.legend()
        ax.grid(True)
        
        # 3D trajectory
        ax = axes[2, 0]
        ax = fig.add_subplot(3, 2, 5, projection='3d')
        pos = traj['position']
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2)
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='g', s=100, label='Start')
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='r', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        
        # Contact schedule
        ax = axes[2, 1]
        phase_times = np.cumsum([0] + results['phase_durations'])
        for i, contacts in enumerate(results['contact_sequence']):
            t_start = phase_times[i]
            t_end = phase_times[i+1]
            for j, contact in enumerate(contacts):
                if contact:
                    ax.barh(j, t_end - t_start, left=t_start, 
                           height=0.8, alpha=0.7)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['FR', 'FL', 'RR', 'RL'])
        ax.set_xlabel('Time (s)')
        ax.set_xlim([0, time[-1]])
        ax.grid(True, axis='x')
        
        # Add phase boundaries
        for t in phase_times[1:-1]:
            for a in axes.flat:
                if hasattr(a, 'axvline'):
                    a.axvline(x=t, color='k', linestyle=':', alpha=0.5)
        
        plt.suptitle('Contact Timing Optimization Results')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize robot parameters
    robot = RobotParams()
    optimizer = ContactTimingOptimizer(robot)
    
    # # Example 1: Forward jump with takeoff and landing sequence
    # print("Example 1: Forward Jump with Full Contact Sequence")
    # jump_task = JumpTask(
    #     p_initial=np.array([0, 0, 0.33]),
    #     p_final=np.array([2.0, 0, 0.33])
    # )
    
    # contact_sequence = [
    #     [1, 1, 1, 1],  # All feet in contact (stance)
    #     [0, 0, 1, 1],  # Rear feet only (takeoff)
    #     [0, 0, 0, 0],  # Flight
    #     [1, 1, 1, 1]   # All feet landing
    # ]
    
    # results = optimizer.optimize_contact_timing(
    #     contact_sequence=contact_sequence,
    #     jump_task=jump_task,
    #     T_bounds=(0.4, 2.0),
    #     N_per_phase=15
    # )
    
    # if results['success']:
    #     # Get reference trajectory in 3xN format
    #     ref_traj_3xN, time_grid = optimizer.get_reference_trajectory_3xN(results)
        
    #     print("\n" + "="*50)
    #     print("REFERENCE TRAJECTORY FOR FULL-BODY OPTIMIZATION")
    #     print("="*50)
    #     print(f"\nTrajectory shape: {ref_traj_3xN.shape}")
    #     print(f"Time grid length: {len(time_grid)}")
    #     print(f"Time range: [{time_grid[0]:.3f}, {time_grid[-1]:.3f}] seconds")
        
    #     # Get contact events
    #     events = optimizer.get_contact_events(results)
    #     print("\nKey Contact Events:")
    #     for event_name, event_time in events.items():
    #         print(f"  {event_name}: {event_time:.3f} s")
        
    #     # Calculate flight duration
    #     if 'takeoff_time' in events and 'landing_time' in events:
    #         flight_duration = events['landing_time'] - events['takeoff_time']
    #         print(f"\nFlight duration: {flight_duration:.3f} s")
        
    #     # Save to file if needed
    #     np.savetxt('reference_trajectory.txt', ref_traj_3xN, 
    #                fmt='%.6f', header='X, Y, Z positions (each row)')
    #     print("\nReference trajectory saved to 'reference_trajectory.txt'")
        
    #     # Plot results
    #     optimizer.plot_results(results)
    
    # # Example 2: Vertical jump with landing
    # print("\n\nExample 2: Vertical Jump with Landing")
    # jump_task_vertical = JumpTask(
    #     p_initial=np.array([0, 0, 0.3]),
    #     p_final=np.array([0, 0, 0.3])  # Return to same height
    # )
    
    # contact_sequence_vertical = [
    #     [1, 1, 1, 1],  # All feet (stance)
    #     [0, 0, 0, 0],  # Flight
    #     [1, 1, 1, 1]   # All feet (landing)
    # ]
    
    # results_vertical = optimizer.optimize_contact_timing(
    #     contact_sequence=contact_sequence_vertical,
    #     jump_task=jump_task_vertical,
    #     T_bounds=(0.3, 1.5),
    #     N_per_phase=20
    # )
    
    # if results_vertical['success']:
    #     # Get reference trajectory
    #     ref_traj_vert, time_vert = optimizer.get_reference_trajectory_3xN(results_vertical)
        
    #     print("\n" + "="*50)
    #     print("VERTICAL JUMP REFERENCE TRAJECTORY")
    #     print("="*50)
    #     print(f"Trajectory shape: {ref_traj_vert.shape}")
        
    #     # Get contact events
    #     events_vert = optimizer.get_contact_events(results_vertical)
    #     print("\nContact Events:")
    #     print(f"  Takeoff: {events_vert.get('takeoff_time', 'N/A'):.3f} s")
    #     print(f"  Landing: {events_vert.get('landing_time', 'N/A'):.3f} s")
    #     print(f"  Flight duration: {events_vert.get('landing_time', 0) - events_vert.get('takeoff_time', 0):.3f} s")
        
    #     optimizer.plot_results(results_vertical)
    
    # Example 3: Complex jump with asymmetric landing
    print("\n\nExample 3: Diagonal Jump with Staggered Landing")
    jump_task_complex = JumpTask(
        p_initial=np.array([0, 0, 0.33]),
        p_final=np.array([2.0, 0, 0.33])  # Diagonal jump
    )
    
    contact_sequence_complex = [
        [1, 1, 1, 1],  # All feet (stance)
        [0, 0, 1, 1],  # Rear feet only (takeoff prep)
        [0, 0, 0, 0],  # Flight
        [1, 1, 0, 0],  # Front feet land first
        [1, 1, 1, 1]   # All feet (full landing)
    ]
    
    results_complex = optimizer.optimize_contact_timing(
        contact_sequence=contact_sequence_complex,
        jump_task=jump_task_complex,
        T_bounds=(0.5, 2.5),
        N_per_phase=12
    )
    
    if results_complex['success']:
        print("\n" + "="*50)
        print("COMPLEX JUMP TRAJECTORY")
        print("="*50)
        
        # Show detailed phase breakdown
        print("\nDetailed Phase Breakdown:")
        cumulative_time = 0.0
        for i, (duration, contacts) in enumerate(zip(results_complex['phase_durations'], 
                                                     results_complex['contact_sequence'])):
            contact_str = ['FR', 'FL', 'RR', 'RL']
            active_feet = [contact_str[j] for j, c in enumerate(contacts) if c]
            if not active_feet:
                active_feet = ['FLIGHT']
            
            print(f"  Phase {i+1}: t=[{cumulative_time:.3f}, {cumulative_time + duration:.3f}]s")
            print(f"           Duration: {duration*1000:.1f}ms")
            print(f"           Active feet: {', '.join(active_feet)}")
            
            cumulative_time += duration
        
        optimizer.plot_results(results_complex)