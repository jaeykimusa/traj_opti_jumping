"""
Contact-timing and Trajectory Optimization for 3D Jumping on Quadruped Robots
Fixed version with proper phase timing distribution
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
        
        # Optimization weights - adjusted for better phase distribution
        self.weights = {
            'omega': 1e-2,      # Angular velocity
            'force': 1e-4,      # Ground reaction forces (reduced)
            'rotation': 1e-1,   # Rotation error
            'height': 10.0,     # Reward for achieving height
            'distance': 5.0,    # Reward for achieving distance
            'phase_penalty': 100.0  # Penalty for too short phases
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
    
    def estimate_flight_time(self, jump_distance, jump_height=0):
        """Estimate required flight time based on jump distance and height"""
        # For horizontal jump: t_flight ≈ distance / average_velocity
        # Assuming average horizontal velocity of 2-3 m/s
        avg_velocity = 2.5
        t_flight = jump_distance / avg_velocity
        
        # For vertical component
        if jump_height > 0:
            # t_up = sqrt(2*h/g)
            t_up = np.sqrt(2 * jump_height / self.robot.gravity)
            t_flight = max(t_flight, 2 * t_up)
        
        return np.clip(t_flight, 0.3, 1.5)  # Reasonable bounds
    
    def optimize_contact_timing(self, 
                              contact_sequence: List[List[int]], 
                              jump_task: JumpTask,
                              T_bounds: Tuple[float, float] = (0.3, 2.5),
                              N_per_phase: int = 15) -> Dict:
        """
        Main optimization function that solves for optimal contact timings.
        """
        
        n_phases = len(contact_sequence)
        N_total = n_phases * N_per_phase
        
        # Estimate jump parameters
        jump_distance = np.linalg.norm(jump_task.p_final[:2] - jump_task.p_initial[:2])
        jump_height = max(0, jump_task.p_final[2] - jump_task.p_initial[2])
        estimated_flight_time = self.estimate_flight_time(jump_distance, jump_height)
        
        # Create optimization problem
        opti = ca.Opti()
        
        # ============ Decision Variables ============
        
        # Phase durations with phase-specific bounds
        T_phases = []
        for i in range(n_phases):
            T_i = opti.variable()
            
            # Set phase-specific bounds based on contact type
            if sum(contact_sequence[i]) == 4:  # Full contact (stance/landing)
                opti.subject_to(0.1 <= T_i)   # Min 100ms
                opti.subject_to(T_i <= 0.5)    # Max 500ms
            elif sum(contact_sequence[i]) == 2:  # Partial contact (rear feet)
                opti.subject_to(0.05 <= T_i)   # Min 50ms
                opti.subject_to(T_i <= 0.3)    # Max 300ms
            elif sum(contact_sequence[i]) == 0:  # Flight
                opti.subject_to(0.2 <= T_i)    # Min 200ms for flight
                opti.subject_to(T_i <= 1.5)    # Max 1.5s
            
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
        opti.subject_to(states[-1]['p'][0] == jump_task.p_final[0])  # X position
        opti.subject_to(states[-1]['p'][1] == jump_task.p_final[1])  # Y position
        # Z position is soft constraint through cost
        
        # ============ Cost Function ============
        cost = 0
        
        # Add phase duration penalties to encourage proper distribution
        for i in range(n_phases):
            if sum(contact_sequence[i]) == 0:  # Flight phase
                # Penalize flight phase being too short
                cost += self.weights['phase_penalty'] * ((estimated_flight_time - T_phases[i])**2)
        
        # ============ Dynamics Constraints ============
        k_global = 0
        max_z = states[0]['p'][2]  # Initialize max height tracker
        
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
                    
                    # Track maximum height for reward
                    max_z = ca.fmax(max_z, p[2])
                    
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
                    
                    # Ground constraint during contact
                    if sum(contacts) > 0:
                        # Penalize foot being below ground during contact
                        ground_violation = ca.fmin(0, p[2] - jump_task.p_initial[2] + 0.05)
                        cost += 1000 * ground_violation**2
                
            k_global += N_per_phase
        
        # Terminal costs
        p_final = states[-1]['p']
        v_final = states[-1]['v']
        
        # Reward for achieving target position
        cost += self.weights['distance'] * ((p_final[0] - jump_task.p_final[0])**2 + 
                                           (p_final[1] - jump_task.p_final[1])**2)
        
        # Reward for achieving height
        cost += -self.weights['height'] * max_z
        
        # Soft landing (low final velocity)
        cost += 10.0 * ca.dot(v_final, v_final)
        
        # Final height constraint (soft)
        cost += 50.0 * (p_final[2] - jump_task.p_final[2])**2
        
        # ============ Solve Optimization ============
        opti.minimize(cost)
        
        # Solver settings
        opts = {
            'ipopt.print_level': 3,
            'ipopt.max_iter': 5000,
            'ipopt.tol': 1e-5,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes'
        }
        opti.solver('ipopt', opts)
        
        # Better initial guess
        self._set_improved_initial_guess(opti, T_phases, states, controls, 
                                       contact_sequence, jump_task, N_per_phase,
                                       estimated_flight_time)
        
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
    
    def _set_improved_initial_guess(self, opti, T_phases, states, controls, 
                                  contact_sequence, jump_task, N_per_phase,
                                  estimated_flight_time):
        """Set improved initial guess for optimization"""
        
        n_phases = len(contact_sequence)
        
        # Better initial guess for phase durations based on contact type
        phase_types = []
        initial_durations = []
        
        for i, contacts in enumerate(contact_sequence):
            if sum(contacts) == 4:
                phase_types.append('stance')
                initial_durations.append(0.3)  # 300ms for stance
            elif sum(contacts) == 2:
                phase_types.append('partial')
                initial_durations.append(0.15)  # 150ms for partial contact
            elif sum(contacts) == 0:
                phase_types.append('flight')
                initial_durations.append(estimated_flight_time)  # Estimated flight time
                
        # Set initial values for phase durations
        for i in range(n_phases):
            opti.set_initial(T_phases[i], initial_durations[i])
        
        # Calculate phase start times using initial durations
        phase_start_times = [0]
        for i in range(n_phases):
            phase_start_times.append(phase_start_times[-1] + initial_durations[i])
        
        # Find flight phase
        flight_phase_idx = None
        for i, contacts in enumerate(contact_sequence):
            if sum(contacts) == 0:
                flight_phase_idx = i
                break
        
        # Initial guess for states with parabolic trajectory during flight
        N_total = n_phases * N_per_phase
        k = 0
        
        for phase_idx in range(n_phases):
            t_phase_start = phase_start_times[phase_idx]
            t_phase_end = phase_start_times[phase_idx + 1]
            
            for k_local in range(N_per_phase if phase_idx < n_phases - 1 else N_per_phase + 1):
                if k <= N_total:
                    t = t_phase_start + k_local * (t_phase_end - t_phase_start) / N_per_phase
                    
                    if flight_phase_idx is None or phase_idx < flight_phase_idx:
                        # Before flight - stay at initial position
                        p_guess = jump_task.p_initial
                        v_guess = [0, 0, 0]
                    elif phase_idx == flight_phase_idx:
                        # During flight - parabolic trajectory
                        flight_progress = k_local / N_per_phase
                        
                        # Horizontal motion (constant velocity)
                        v_horizontal = (jump_task.p_final[:2] - jump_task.p_initial[:2]) / estimated_flight_time
                        p_x = jump_task.p_initial[0] + v_horizontal[0] * flight_progress * estimated_flight_time
                        p_y = jump_task.p_initial[1] + v_horizontal[1] * flight_progress * estimated_flight_time
                        
                        # Vertical motion (parabolic)
                        t_flight = flight_progress * estimated_flight_time
                        v0_z = estimated_flight_time * self.robot.gravity / 2  # Initial vertical velocity
                        p_z = jump_task.p_initial[2] + v0_z * t_flight - 0.5 * self.robot.gravity * t_flight**2
                        
                        p_guess = [p_x, p_y, max(p_z, 0.1)]
                        v_guess = [v_horizontal[0], v_horizontal[1], v0_z - self.robot.gravity * t_flight]
                    else:
                        # After flight - at final position
                        p_guess = jump_task.p_final
                        v_guess = [0, 0, 0]
                    
                    opti.set_initial(states[k]['p'], p_guess)
                    opti.set_initial(states[k]['v'], v_guess)
                    opti.set_initial(states[k]['q'], [1, 0, 0, 0])
                    opti.set_initial(states[k]['omega'], [0, 0, 0])
                    
                    k += 1
        
        # Initial guess for forces
        k = 0
        for phase_idx in range(n_phases):
            contacts = contact_sequence[phase_idx]
            
            for k_local in range(N_per_phase):
                if k < N_total:
                    for i, foot_name in enumerate(['FR', 'FL', 'RR', 'RL']):
                        if contacts[i]:
                            # Distribute weight among contact feet
                            n_contacts = sum(contacts)
                            f_z = self.robot.mass * self.robot.gravity / n_contacts
                            
                            # Add some forward force during takeoff phase
                            if flight_phase_idx and phase_idx < flight_phase_idx and k_local > N_per_phase // 2:
                                f_x = 20.0  # Forward push
                            else:
                                f_x = 0
                                
                            opti.set_initial(controls[k][foot_name], [f_x, 0, f_z])
                        else:
                            opti.set_initial(controls[k][foot_name], [0, 0, 0])
                    k += 1
    
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
        """Get reference trajectory in 3xN format"""
        if not results['success']:
            return None, None
        
        positions = results['trajectory']['position'].T  # Transpose to 3xN
        times = results['trajectory']['time']
        
        return positions, times
    
    def get_contact_events(self, results):
        """Extract contact event timings"""
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
    
    # Example: Forward jump with proper contact sequence
    print("Forward Jump with Optimized Contact Timing")
    jump_task = JumpTask(
        p_initial=np.array([0, 0, 0.33]),
        p_final=np.array([2.0, 0, 0.33])
    )
    
    contact_sequence = [
        [1, 1, 1, 1],  # All feet (stance) - preparation
        [0, 0, 1, 1],  # Rear feet only (takeoff)
        [0, 0, 0, 0],  # Flight
        [1, 1, 0, 0],  # Front feet land first
        [1, 1, 1, 1]   # All feet (full landing)
    ]
    
    results = optimizer.optimize_contact_timing(
        contact_sequence=contact_sequence,
        jump_task=jump_task,
        T_bounds=(0.8, 2.5),
        N_per_phase=12
    )
    
    if results['success']:
        # Get reference trajectory in 3xN format
        ref_traj_3xN, time_grid = optimizer.get_reference_trajectory_3xN(results)
        
        print("\n" + "="*50)
        print("REFERENCE TRAJECTORY FOR FULL-BODY OPTIMIZATION")
        print("="*50)
        print(f"\nTrajectory shape: {ref_traj_3xN.shape}")
        print(f"Time grid length: {len(time_grid)}")
        print(f"Time range: [{time_grid[0]:.3f}, {time_grid[-1]:.3f}] seconds")
        
        # Get contact events
        events = optimizer.get_contact_events(results)
        print("\nKey Contact Events:")
        for event_name, event_time in events.items():
            print(f"  {event_name}: {event_time:.3f} s")
        
        # Calculate flight duration
        if 'takeoff_time' in events and 'landing_time' in events:
            flight_duration = events['landing_time'] - events['takeoff_time']
            print(f"\nFlight duration: {flight_duration:.3f} s")
        
        # Show phase timing breakdown
        print("\n" + "="*50)
        print("DETAILED PHASE TIMING ANALYSIS")
        print("="*50)
        cumulative_time = 0.0
        for i, (duration, contacts) in enumerate(zip(results['phase_durations'], 
                                                     results['contact_sequence'])):
            contact_str = ['FR', 'FL', 'RR', 'RL']
            active_feet = [contact_str[j] for j, c in enumerate(contacts) if c]
            if not active_feet:
                active_feet = ['FLIGHT']
            
            percentage = (duration / results['total_time']) * 100
            
            print(f"\nPhase {i+1}:")
            print(f"  Time interval: [{cumulative_time:.3f}, {cumulative_time + duration:.3f}] s")
            print(f"  Duration: {duration*1000:.1f} ms ({percentage:.1f}% of total)")
            print(f"  Active feet: {', '.join(active_feet)}")
            
            # Phase-specific analysis
            if sum(contacts) == 0:  # Flight phase
                # Calculate max height during this phase
                phase_start_idx = int(cumulative_time / results['total_time'] * len(time_grid))
                phase_end_idx = int((cumulative_time + duration) / results['total_time'] * len(time_grid))
                phase_positions = ref_traj_3xN[:, phase_start_idx:phase_end_idx]
                if phase_positions.size > 0:
                    max_flight_height = np.max(phase_positions[2, :])
                    print(f"  Max height during flight: {max_flight_height:.3f} m")
            
            cumulative_time += duration
        
        # Save to file
        np.savetxt('reference_trajectory.txt', ref_traj_3xN, 
                   fmt='%.6f', header='X, Y, Z positions (each row)')
        print("\nReference trajectory saved to 'reference_trajectory.txt'")
        
        # Plot results
        optimizer.plot_results(results)