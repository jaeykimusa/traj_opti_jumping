"""
Contact-timing and Trajectory Optimization for 3D Jumping on Quadruped Robots
Fixed version with proper rotation representation and constraints
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

@dataclass
class RobotParams:
    """Robot parameters for Go2 quadruped"""
    mass: float = 15.0  # kg
    body_length: float = 0.71  # m
    body_width: float = 0.31  # m
    body_height: float = 0.40  # m
    gravity: float = 9.81  # m/s^2
    inertia: np.ndarray = None  # Body inertia tensor
    mu: float = 0.6  # Friction coefficient
    f_min: float = 0.0  # Min normal force
    f_max: float = 150.0  # Max normal force per foot
    
    def __post_init__(self):
        if self.inertia is None:
            # Default inertia for Go2 robot
            I_xx = (1/12) * self.mass * (self.body_height**2 + self.body_width**2)
            I_yy = (1/12) * self.mass * (self.body_length**2 + self.body_height**2)
            I_zz = (1/12) * self.mass * (self.body_length**2 + self.body_width**2)
            self.inertia = np.diag([I_xx, I_yy, I_zz])

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
    Following the approach from the paper by Nguyen & Nguyen (2022)
    """
    
    def __init__(self):
        self.robot = RobotParams()
        
        # Foot positions in body frame (matching paper's A1 robot)
        self.r_foot_body = {
            'FR': np.array([0.183, -0.13, 0]),
            'FL': np.array([0.183, 0.13, 0]),
            'RR': np.array([-0.183, -0.13, 0]),
            'RL': np.array([-0.183, 0.13, 0])
        }
        
        # Optimization weights
        self.weights = {
            'omega': 1e-2,      # Angular velocity
            'force': 1e-4,      # Ground reaction forces
            'rotation': 1e-1,   # Rotation error
            'height': 10.0,     # Reward for achieving height
            'distance': 5.0,    # Reward for achieving distance
            'phase_penalty': 100.0  # Penalty for too short phases
        }
    
    def euler_to_rotation_matrix(self, euler):
        """Convert Euler angles to rotation matrix (ZYX convention)"""
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        # Rotation matrices for each axis
        cx = ca.cos(roll)
        sx = ca.sin(roll)
        cy = ca.cos(pitch)
        sy = ca.sin(pitch)
        cz = ca.cos(yaw)
        sz = ca.sin(yaw)
        
        # Combined rotation: R = Rz * Ry * Rx
        R = ca.vertcat(
            ca.horzcat(cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx),
            ca.horzcat(sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx),
            ca.horzcat(-sy, cy*sx, cy*cx)
        )
        
        return R
    
    def optimize_contact_timing(self, 
                              contact_sequence: List[List[int]], 
                              jump_task: JumpTask,
                              T_bounds: Tuple[float, float] = (0.3, 2.5),
                              N_per_phase: int = 15) -> Dict:
        """
        Main optimization function that solves for optimal contact timings.
        Returns generalized coordinates [p; orientation] as 6xN matrix.
        """
        
        n_phases = len(contact_sequence)
        N_total = n_phases * N_per_phase
        
        # Create optimization problem
        opti = ca.Opti()
        
        # ============ Decision Variables ============
        
        # Phase durations with phase-specific bounds
        T_phases = []
        for i in range(n_phases):
            T_i = opti.variable()
            
            # Set phase-specific bounds based on contact type
            if sum(contact_sequence[i]) == 4:  # Full contact
                opti.subject_to(0.1 <= T_i)
                opti.subject_to(T_i <= 0.5)
            elif sum(contact_sequence[i]) == 2:  # Partial contact
                opti.subject_to(0.05 <= T_i)
                opti.subject_to(T_i <= 0.3)
            elif sum(contact_sequence[i]) == 0:  # Flight
                opti.subject_to(0.2 <= T_i)
                opti.subject_to(T_i <= 1.5)
            
            T_phases.append(T_i)
        
        # Total time constraint
        T_total = sum(T_phases)
        opti.subject_to(T_bounds[0] <= T_total)
        opti.subject_to(T_total <= T_bounds[1])
        
        # State trajectory variables (using Euler angles for simplicity)
        states = []
        for k in range(N_total + 1):
            state = {
                'p': opti.variable(3),        # Position
                'v': opti.variable(3),        # Velocity
                'euler': opti.variable(3),    # Euler angles (roll, pitch, yaw)
                'omega': opti.variable(3)     # Angular velocity (body frame)
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
        # Convert initial rotation matrix to Euler angles
        initial_euler = self._rotation_matrix_to_euler_numpy(jump_task.R_initial)
        
        opti.subject_to(states[0]['p'] == jump_task.p_initial)
        opti.subject_to(states[0]['v'] == 0)
        opti.subject_to(states[0]['euler'] == initial_euler)
        opti.subject_to(states[0]['omega'] == 0)
        
        # ============ Final Conditions ============
        final_euler = self._rotation_matrix_to_euler_numpy(jump_task.R_final)
        
        opti.subject_to(states[-1]['p'][0] == jump_task.p_final[0])
        opti.subject_to(states[-1]['p'][1] == jump_task.p_final[1])
        # Soft constraint for final orientation
        
        # ============ Cost Function ============
        cost = 0
        
        # Add phase duration penalties
        jump_distance = np.linalg.norm(jump_task.p_final[:2] - jump_task.p_initial[:2])
        jump_height = max(0, jump_task.p_final[2] - jump_task.p_initial[2])
        estimated_flight_time = self.estimate_flight_time(jump_distance, jump_height)
        
        for i in range(n_phases):
            if sum(contact_sequence[i]) == 0:  # Flight phase
                cost += self.weights['phase_penalty'] * ((estimated_flight_time - T_phases[i])**2)
        
        # ============ Dynamics Constraints ============
        k_global = 0
        max_z = states[0]['p'][2]
        
        for phase_idx in range(n_phases):
            dt = T_phases[phase_idx] / N_per_phase
            contacts = contact_sequence[phase_idx]
            
            for k_local in range(N_per_phase):
                k = k_global + k_local
                
                if k < N_total:
                    # Current state
                    p = states[k]['p']
                    v = states[k]['v']
                    euler = states[k]['euler']
                    omega = states[k]['omega']
                    
                    # Get rotation matrix from Euler angles
                    R = self.euler_to_rotation_matrix(euler)
                    
                    max_z = ca.fmax(max_z, p[2])
                    
                    # Total force and torque
                    F_total = ca.MX([0, 0, -self.robot.mass * self.robot.gravity])
                    tau_total = ca.MX.zeros(3, 1)
                    
                    # Process each foot
                    for foot_name, foot_pos_body in self.r_foot_body.items():
                        foot_idx = ['FR', 'FL', 'RR', 'RL'].index(foot_name)
                        
                        if contacts[foot_idx]:  # Foot in contact
                            f = controls[k][foot_name]
                            F_total += f
                            
                            # Foot position in world frame
                            r_foot = R @ foot_pos_body
                            tau_total += ca.cross(r_foot, f)
                            
                            # Force constraints
                            opti.subject_to(0 <= f[2])
                            opti.subject_to(f[2] <= self.robot.f_max)
                            
                            # Friction cone
                            opti.subject_to(f[0]**2 + f[1]**2 <= (self.robot.mu * f[2])**2)
                        else:
                            opti.subject_to(controls[k][foot_name] == 0)
                    
                    # Translation dynamics: p_dot = v, v_dot = F/m
                    p_next = p + dt * v
                    v_next = v + dt * F_total / self.robot.mass
                    
                    # Euler angle dynamics (simplified)
                    # Convert body angular velocity to Euler angle rates
                    roll, pitch, yaw = euler[0], euler[1], euler[2]
                    
                    # Transformation matrix from body rates to Euler rates
                    # euler_dot = T * omega_body
                    cr = ca.cos(roll)
                    sr = ca.sin(roll)
                    cp = ca.cos(pitch)
                    sp = ca.sin(pitch)
                    
                    # Avoid singularity at pitch = ±90 degrees
                    cp_safe = ca.fmax(cp, 0.01)
                    
                    T_inv = ca.vertcat(
                        ca.horzcat(1, sr*sp/cp_safe, cr*sp/cp_safe),
                        ca.horzcat(0, cr, -sr),
                        ca.horzcat(0, sr/cp_safe, cr/cp_safe)
                    )
                    
                    euler_dot = T_inv @ omega
                    euler_next = euler + dt * euler_dot
                    
                    # Angular dynamics: I*omega_dot = tau - omega x (I*omega)
                    I_omega = self.robot.inertia @ omega
                    omega_cross_I_omega = ca.cross(omega, I_omega)
                    tau_body = R.T @ tau_total
                    omega_dot = ca.solve(self.robot.inertia, tau_body - omega_cross_I_omega)
                    omega_next = omega + dt * omega_dot
                    
                    # Apply dynamics constraints
                    opti.subject_to(states[k+1]['p'] == p_next)
                    opti.subject_to(states[k+1]['v'] == v_next)
                    opti.subject_to(states[k+1]['euler'] == euler_next)
                    opti.subject_to(states[k+1]['omega'] == omega_next)
                    
                    # Add costs
                    cost += self.weights['omega'] * ca.dot(omega, omega)
                    for foot_name in ['FR', 'FL', 'RR', 'RL']:
                        f = controls[k][foot_name]
                        cost += self.weights['force'] * ca.dot(f, f)
                    
                    # Orientation error cost
                    euler_error = euler - final_euler
                    cost += self.weights['rotation'] * ca.dot(euler_error, euler_error)
                    
                    # Ground constraint during contact
                    if sum(contacts) > 0:
                        # Penalize being too low during contact
                        ground_violation = ca.fmin(0, p[2] - 0.1)
                        cost += 1000 * ground_violation**2
                
            k_global += N_per_phase
        
        # Terminal costs
        p_final = states[-1]['p']
        v_final = states[-1]['v']
        euler_final = states[-1]['euler']
        
        cost += self.weights['distance'] * ((p_final[0] - jump_task.p_final[0])**2 + 
                                           (p_final[1] - jump_task.p_final[1])**2)
        cost += -self.weights['height'] * max_z
        cost += 10.0 * ca.dot(v_final, v_final)
        cost += 50.0 * (p_final[2] - jump_task.p_final[2])**2
        
        # Final orientation cost
        euler_error_final = euler_final - final_euler
        cost += 100.0 * ca.dot(euler_error_final, euler_error_final)
        
        # ============ Solve Optimization ============
        opti.minimize(cost)
        
        # Solver settings
        opts = {
            'ipopt.print_level': 3,
            'ipopt.max_iter': 3000,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes'
        }
        opti.solver('ipopt', opts)
        
        # Set initial guess
        self._set_initial_guess(opti, T_phases, states, controls, 
                               contact_sequence, jump_task, N_per_phase,
                               initial_euler, final_euler)
        
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
            
            # Extract generalized coordinates (6xN: position + orientation)
            generalized_coords = np.zeros((6, N_total + 1))
            
            for k in range(N_total + 1):
                # Position (first 3 rows)
                p_k = sol.value(states[k]['p'])
                generalized_coords[0:3, k] = p_k
                
                # Orientation as Euler angles (last 3 rows)
                euler_k = sol.value(states[k]['euler'])
                generalized_coords[3:6, k] = euler_k
            
            # Extract full trajectory for compatibility
            trajectory = {
                'time': np.array(time_grid),
                'position': generalized_coords[0:3, :].T,  # Nx3 format
                'velocity': np.array([sol.value(states[k]['v']) for k in range(N_total + 1)]),
                'euler_angles': generalized_coords[3:6, :].T,  # Nx3 format
                'omega': np.array([sol.value(states[k]['omega']) for k in range(N_total + 1)]),
                'forces': {}
            }
            
            # Extract forces
            for foot_name in ['FR', 'FL', 'RR', 'RL']:
                trajectory['forces'][foot_name] = np.array([
                    sol.value(controls[k][foot_name]) if k < N_total else np.zeros(3)
                    for k in range(N_total + 1)
                ])
            
            # Results summary
            results = {
                'success': True,
                'phase_durations': T_opt,
                'total_time': sum(T_opt),
                'generalized_coordinates': generalized_coords,  # 6xN matrix
                'trajectory': trajectory,
                'contact_sequence': contact_sequence,
                'cost': sol.value(cost),
                'time_grid': np.array(time_grid)
            }
            
            # Print summary
            self._print_results_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _rotation_matrix_to_euler_numpy(self, R):
        """Convert rotation matrix to Euler angles (numpy version for post-processing)"""
        # Using ZYX convention
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])
    
    def estimate_flight_time(self, jump_distance, jump_height=0):
        """Estimate required flight time based on jump distance and height"""
        avg_velocity = 2.5
        t_flight = jump_distance / avg_velocity
        
        if jump_height > 0:
            t_up = np.sqrt(2 * jump_height / self.robot.gravity)
            t_flight = max(t_flight, 2 * t_up)
        
        return np.clip(t_flight, 0.3, 2.0)
    
    def _set_initial_guess(self, opti, T_phases, states, controls, 
                           contact_sequence, jump_task, N_per_phase,
                           initial_euler, final_euler):
        """Set initial guess for optimization"""
        
        n_phases = len(contact_sequence)
        
        # Initial guess for phase durations
        initial_durations = []
        for i, contacts in enumerate(contact_sequence):
            if sum(contacts) == 4:
                initial_durations.append(0.3)
            elif sum(contacts) == 2:
                initial_durations.append(0.15)
            elif sum(contacts) == 0:
                jump_distance = np.linalg.norm(jump_task.p_final[:2] - jump_task.p_initial[:2])
                initial_durations.append(self.estimate_flight_time(jump_distance))
        
        for i in range(n_phases):
            opti.set_initial(T_phases[i], initial_durations[i])
        
        # Initial guess for states
        N_total = n_phases * N_per_phase
        for k in range(N_total + 1):
            # Linear interpolation for position
            alpha = k / N_total
            p_guess = (1 - alpha) * jump_task.p_initial + alpha * jump_task.p_final
            
            # Linear interpolation for orientation
            euler_guess = (1 - alpha) * initial_euler + alpha * final_euler
            
            opti.set_initial(states[k]['p'], p_guess)
            opti.set_initial(states[k]['v'], [0, 0, 0])
            opti.set_initial(states[k]['euler'], euler_guess)
            opti.set_initial(states[k]['omega'], [0, 0, 0])
        
        # Initial guess for forces
        for k in range(N_total):
            for foot_name in ['FR', 'FL', 'RR', 'RL']:
                opti.set_initial(controls[k][foot_name], [0, 0, self.robot.mass * self.robot.gravity / 4])
    
    def _print_results_summary(self, results):
        """Print optimization results summary"""
        print("\n" + "="*50)
        print("CONTACT TIMING OPTIMIZATION RESULTS")
        print("="*50)
        
        print(f"\nTotal time: {results['total_time']:.3f} s")
        
        # Print generalized coordinates info
        gen_coords = results['generalized_coordinates']
        print(f"\nGeneralized Coordinates (6 x {gen_coords.shape[1]}):")
        print("  Rows 0-2: CoM position (x, y, z)")
        print("  Rows 3-5: Body orientation (roll, pitch, yaw)")
        
        # Print sample values
        print("\n  First 5 timesteps:")
        labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        for i, label in enumerate(labels):
            values = gen_coords[i, :min(5, gen_coords.shape[1])]
            if i >= 3:  # Convert radians to degrees for angles
                values = values * 180/np.pi
                print(f"    {label:5s}: [{', '.join(f'{v:.1f}°' for v in values)} ...]")
            else:
                print(f"    {label:5s}: [{', '.join(f'{v:.3f}' for v in values)} ...]")
        
        print("\n  Last 5 timesteps:")
        for i, label in enumerate(labels):
            values = gen_coords[i, -min(5, gen_coords.shape[1]):]
            if i >= 3:  # Convert radians to degrees for angles
                values = values * 180/np.pi
                print(f"    {label:5s}: [... {', '.join(f'{v:.1f}°' for v in values)}]")
            else:
                print(f"    {label:5s}: [... {', '.join(f'{v:.3f}' for v in values)}]")
        
        # Phase durations
        print("\nPhase durations:")
        for i, (duration, contacts) in enumerate(zip(results['phase_durations'], 
                                                     results['contact_sequence'])):
            contact_str = ['FR', 'FL', 'RR', 'RL']
            active_feet = [contact_str[j] for j, c in enumerate(contacts) if c]
            if not active_feet:
                active_feet = ['FLIGHT']
            print(f"  Phase {i+1} ({', '.join(active_feet)}): {duration*1000:.1f} ms")
        
        # Trajectory statistics
        positions = gen_coords[0:3, :]
        orientations = gen_coords[3:6, :]
        
        max_height = np.max(positions[2, :])
        max_roll = np.max(np.abs(orientations[0, :])) * 180/np.pi
        max_pitch = np.max(np.abs(orientations[1, :])) * 180/np.pi
        max_yaw = np.max(np.abs(orientations[2, :])) * 180/np.pi
        
        print(f"\nTrajectory statistics:")
        print(f"  Max height: {max_height:.3f} m")
        print(f"  Jump distance: {positions[0, -1] - positions[0, 0]:.3f} m")
        print(f"  Max roll: {max_roll:.1f} deg")
        print(f"  Max pitch: {max_pitch:.1f} deg")
        print(f"  Max yaw: {max_yaw:.1f} deg")
        
        print(f"\nOptimization cost: {results['cost']:.6f}")
    
    def get_generalized_coordinates(self, results):
        """
        Get generalized coordinates in 6xN format as per paper.
        Returns: (generalized_coords, time_grid)
        - generalized_coords: 6xN matrix [position; orientation]
        - time_grid: time stamps for each point
        """
        if not results['success']:
            return None, None
        
        return results['generalized_coordinates'], results['time_grid']
    
    def plot_results(self, results):
        """Visualize optimization results including orientation"""
        if not results['success']:
            print("Cannot plot failed optimization")
            return
        
        gen_coords = results['generalized_coordinates']
        time = results['time_grid']
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # Position
        ax = axes[0, 0]
        ax.plot(time, gen_coords[0, :], 'r-', label='x')
        ax.plot(time, gen_coords[1, :], 'g-', label='y')
        ax.plot(time, gen_coords[2, :], 'b-', label='z')
        ax.set_ylabel('Position (m)')
        ax.set_title('CoM Position')
        ax.legend()
        ax.grid(True)
        
        # Orientation (Euler angles)
        ax = axes[0, 1]
        ax.plot(time, gen_coords[3, :] * 180/np.pi, 'r-', label='roll')
        ax.plot(time, gen_coords[4, :] * 180/np.pi, 'g-', label='pitch')
        ax.plot(time, gen_coords[5, :] * 180/np.pi, 'b-', label='yaw')
        ax.set_ylabel('Orientation (deg)')
        ax.set_title('Body Orientation (Euler Angles)')
        ax.legend()
        ax.grid(True)
        
        # Velocity
        traj = results['trajectory']
        ax = axes[1, 0]
        ax.plot(time, traj['velocity'][:, 0], 'r-', label='vx')
        ax.plot(time, traj['velocity'][:, 1], 'g-', label='vy')
        ax.plot(time, traj['velocity'][:, 2], 'b-', label='vz')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('CoM Velocity')
        ax.legend()
        ax.grid(True)
        
        # Angular velocity
        ax = axes[1, 1]
        ax.plot(time, traj['omega'][:, 0], 'r-', label='ωx')
        ax.plot(time, traj['omega'][:, 1], 'g-', label='ωy')
        ax.plot(time, traj['omega'][:, 2], 'b-', label='ωz')
        ax.set_ylabel('Angular velocity (rad/s)')
        ax.set_title('Body Angular Velocity')
        ax.legend()
        ax.grid(True)
        
        # 3D trajectory
        ax = fig.add_subplot(3, 2, 5, projection='3d')
        pos = gen_coords[0:3, :]
        ax.plot(pos[0, :], pos[1, :], pos[2, :], 'b-', linewidth=2)
        ax.scatter(pos[0, 0], pos[1, 0], pos[2, 0], c='g', s=100, label='Start')
        ax.scatter(pos[0, -1], pos[1, -1], pos[2, -1], c='r', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
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
        ax.set_title('Contact Schedule')
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
    import time as timer
    
    start_time = timer.perf_counter()
    
    optimizer = ContactTimingOptimizer()
    
    # Example: Forward jump with proper contact sequence
    print("Forward Jump with Optimized Contact Timing")
    print("Target: 2m forward jump maintaining orientation")
    
    jump_task = JumpTask(
        p_initial=np.array([0, 0, 0.33]),
        p_final=np.array([2.0, 0, 0.33]),
        R_initial=np.eye(3),  # Identity rotation
        R_final=np.eye(3)     # Maintain same orientation
    )
    
    # Contact sequence as in the paper
    contact_sequence = [
        [1, 1, 1, 1],  # All feet (stance) - preparation
        [0, 0, 1, 1],  # Rear feet only (takeoff)
        [0, 0, 0, 0],  # Flight
        [1, 1, 1, 1]   # All feet (full landing)
    ]
    
    print("\nContact sequence:")
    for i, contacts in enumerate(contact_sequence):
        contact_names = ['FR', 'FL', 'RR', 'RL']
        active = [contact_names[j] for j, c in enumerate(contacts) if c]
        print(f"  Phase {i+1}: {active if active else ['FLIGHT']}")
    
    # Run optimization
    results = optimizer.optimize_contact_timing(
        contact_sequence=contact_sequence,
        jump_task=jump_task,
        T_bounds=(0.8, 2.5),
        N_per_phase=12
    )
    
    if results['success']:
        end_time = timer.perf_counter()
        elapsed_time = end_time - start_time
        
        print(f"\nOptimization completed successfully!")
        print(f"Computation time: {elapsed_time:.2f} seconds")
        
        # Get generalized coordinates (6xN format as per paper)
        gen_coords, time_grid = optimizer.get_generalized_coordinates(results)
        
        print("\n" + "="*50)
        print("GENERALIZED COORDINATES OUTPUT")
        print("="*50)
        print(f"\nShape: {gen_coords.shape[0]} x {gen_coords.shape[1]}")
        print("Format: [p_x; p_y; p_z; roll; pitch; yaw]")
        print("\nThis 6xN matrix contains:")
        print("  - Rows 0-2: CoM position trajectory (m)")
        print("  - Rows 3-5: Body orientation trajectory (rad)")
        print(f"  - N = {gen_coords.shape[1]} time points")
        print(f"  - Time span: {time_grid[0]:.3f} to {time_grid[-1]:.3f} seconds")
        
        # Save generalized coordinates to file
        np.savetxt('generalized_coordinates.txt', gen_coords, 
                   fmt='%.6f', 
                   header='Generalized coordinates (6xN): [position(3); orientation(3)] x time_points')
        print("\nGeneralized coordinates saved to 'generalized_coordinates.txt'")
        
        # Also save time grid
        np.savetxt('time_grid.txt', time_grid, 
                   fmt='%.6f',
                   header='Time grid corresponding to generalized coordinates')
        print("Time grid saved to 'time_grid.txt'")
        
        # Example: Access specific components
        print("\n" + "="*50)
        print("EXAMPLE: ACCESSING COMPONENTS")
        print("="*50)
        
        # Extract position and orientation separately
        position_trajectory = gen_coords[0:3, :]  # 3xN
        orientation_trajectory = gen_coords[3:6, :]  # 3xN
        
        print(f"\nPosition trajectory shape: {position_trajectory.shape}")
        print(f"Orientation trajectory shape: {orientation_trajectory.shape}")
        
        # Find key events
        takeoff_idx = int(sum(results['phase_durations'][:2]) / results['total_time'] * len(time_grid))
        landing_idx = int(sum(results['phase_durations'][:3]) / results['total_time'] * len(time_grid))
        
        print(f"\nKey positions:")
        print(f"  Initial: p = [{position_trajectory[0, 0]:.3f}, {position_trajectory[1, 0]:.3f}, {position_trajectory[2, 0]:.3f}]")
        print(f"           θ = [{orientation_trajectory[0, 0]*180/np.pi:.1f}°, {orientation_trajectory[1, 0]*180/np.pi:.1f}°, {orientation_trajectory[2, 0]*180/np.pi:.1f}°]")
        print(f"  Takeoff: p = [{position_trajectory[0, takeoff_idx]:.3f}, {position_trajectory[1, takeoff_idx]:.3f}, {position_trajectory[2, takeoff_idx]:.3f}]")
        print(f"           θ = [{orientation_trajectory[0, takeoff_idx]*180/np.pi:.1f}°, {orientation_trajectory[1, takeoff_idx]*180/np.pi:.1f}°, {orientation_trajectory[2, takeoff_idx]*180/np.pi:.1f}°]")
        print(f"  Landing: p = [{position_trajectory[0, landing_idx]:.3f}, {position_trajectory[1, landing_idx]:.3f}, {position_trajectory[2, landing_idx]:.3f}]")
        print(f"           θ = [{orientation_trajectory[0, landing_idx]*180/np.pi:.1f}°, {orientation_trajectory[1, landing_idx]*180/np.pi:.1f}°, {orientation_trajectory[2, landing_idx]*180/np.pi:.1f}°]")
        print(f"  Final:   p = [{position_trajectory[0, -1]:.3f}, {position_trajectory[1, -1]:.3f}, {position_trajectory[2, -1]:.3f}]")
        print(f"           θ = [{orientation_trajectory[0, -1]*180/np.pi:.1f}°, {orientation_trajectory[1, -1]*180/np.pi:.1f}°, {orientation_trajectory[2, -1]*180/np.pi:.1f}°]")
        
        # Plot results 
        optimizer.plot_results(results) 
        
    else:
        print(f"\nOptimization failed: {results.get('error', 'Unknown error')}")
        print("Try adjusting:")
        print("  - Phase bounds")
        print("  - Number of points per phase")
        print("  - Contact sequence")
        print("  - Jump distance/height targets")