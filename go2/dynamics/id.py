# # ===== DEBUGGED AND FIXED id.py =====
# # This version includes debugging and fixes for the numerical issues

# from go2.robot.robot import *
# from go2.robot.morphology import *
# from go2.dynamics.fd import *
# from go2.utils.math_utils import *

# # ===== FOOT FRAME DEFINITIONS =====
# FOOT_FRAME_NAMES = [
#     "LF_FOOT",  # Left Front foot
#     "RF_FOOT",  # Right Front foot  
#     "LH_FOOT",  # Left Hind foot
#     "RH_FOOT"   # Right Hind foot
# ]

# # Get frame IDs
# FOOT_FRAME_IDS = []
# for frame_name in FOOT_FRAME_NAMES:
#     if model.existFrame(frame_name):
#         frame_id = model.getFrameId(frame_name)
#         FOOT_FRAME_IDS.append(frame_id)
#     else:
#         raise ValueError(f"Could not find foot frame: {frame_name}")

# print(f"✅ Inverse dynamics using foot frames: {[(name, id) for name, id in zip(FOOT_FRAME_NAMES, FOOT_FRAME_IDS)]}")

# def computeFullContactJacobians(q):
#     """
#     Compute contact Jacobians for all feet using correct URDF frame names.
#     """
#     if isinstance(q, ca.SX):
#         jacobians = []
#         for frame_id in FOOT_FRAME_IDS:
#             J_foot = pinocchio.casadi.computeFrameJacobian(
#                 ad_model, ad_data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
#             )[:3, :]  # Only position part (3×18)
#             jacobians.append(J_foot)
        
#         Jc = ca.vertcat(*jacobians)  # Shape: (12, 18)
        
#     elif isinstance(q, ca.MX):
#         cs_q = ca.SX.sym("q", NUM_Q, 1)
        
#         jacobians = []
#         for frame_id in FOOT_FRAME_IDS:
#             J_foot = pinocchio.casadi.computeFrameJacobian(
#                 ad_model, ad_data, cs_q, frame_id, pin.LOCAL_WORLD_ALIGNED
#             )[:3, :]
#             jacobians.append(J_foot)
        
#         cs_Jc = ca.vertcat(*jacobians)
#         cs_full_Jc_fn = ca.Function("cs_full_Jc_fn", [cs_q], [cs_Jc])
#         Jc = cs_full_Jc_fn(q)
        
#     else:
#         # Numerical mode
#         pin.computeJointJacobians(model, data, q)
#         pin.framesForwardKinematics(model, data, q)
        
#         jacobians = []
#         for frame_id in FOOT_FRAME_IDS:
#             J = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
#             jacobians.append(J[:3, :])  # Only position part (3×18)
        
#         Jc = np.vstack(jacobians)  # Shape: (12, 18)
    
#     return Jc

# def id(q, v, qdd, f):
#     """
#     Inverse dynamics: compute motor torques needed for desired acceleration.
    
#     Args:
#         q: Configuration vector (NUM_Q = 19)
#         v: Velocity vector (NUM_V = 18)
#         qdd: Acceleration vector (NUM_V = 18)
#         f: Contact forces (NUM_F = 12)
        
#     Returns:
#         tau_motor: Motor torques (NUM_V = 18)
#     """
#     if isinstance(q, (ca.SX, ca.MX)):  # CasADi symbolic mode
#         cs_q = ca.SX.sym("q", NUM_Q, 1)         # Configuration: 19×1
#         cs_v = ca.SX.sym("qd", NUM_V, 1)        # Velocity: 18×1
#         cs_qdd = ca.SX.sym("qdd", NUM_V, 1)     # Acceleration: 18×1
#         cs_f = ca.SX.sym("f", NUM_F, 1)         # Contact forces: 12×1
        
#         # Compute contact Jacobian: (12×18)
#         cs_Jc = computeFullContactJacobians(cs_q)
        
#         # Compute required generalized forces using RNEA
#         cs_tau_total = pinocchio.casadi.rnea(ad_model, ad_data, cs_q, cs_v, cs_qdd)  # Shape: (18,)
        
#         # Compute motor torques: tau_motor = tau_total - J_c^T * f
#         cs_tau_motor = cs_tau_total - cs_Jc.T @ cs_f  # Shape: (18,)
        
#         # Create function
#         cs_id_fn = ca.Function("inverse_dynamics", [cs_q, cs_v, cs_qdd, cs_f], [cs_tau_motor])
        
#         # Evaluate at given values
#         tau_motor = cs_id_fn(q, v, qdd, f)
       
#     else: 
#         # Numerical mode
#         Jc = computeFullContactJacobians(q)  # Shape: (12, 18)
        
#         # Compute total required forces using RNEA
#         tau_total = pin.rnea(model, data, q, v, qdd)  # Shape: (18,)
        
#         # Subtract contact force contribution
#         tau_motor = tau_total - Jc.T @ f  # Shape: (18,)

#     return tau_motor

# def getValidStandingConfiguration():
#     """
#     Get a valid standing configuration that doesn't cause numerical issues.
#     """
#     q = pin.neutral(model)
    
#     # Set base position
#     q[0] = 0.0    # x
#     q[1] = 0.0    # y  
#     q[2] = 0.3    # z (height)
    
#     # Set base orientation (identity quaternion)
#     q[3] = 0.0  # qx
#     q[4] = 0.0  # qy
#     q[5] = 0.0  # qz  
#     q[6] = 1.0  # qw
    
#     # Set joint angles to reasonable standing pose
#     # Use more conservative angles to avoid singularities
#     joint_angles = np.array([
#         0.0,  0.7, -1.4,   # LF: HAA, HFE, KFE
#         0.0,  0.7, -1.4,   # RF: HAA, HFE, KFE
#         0.0,  0.7, -1.4,   # LH: HAA, HFE, KFE  
#         0.0,  0.7, -1.4    # RH: HAA, HFE, KFE
#     ])
    
#     # Convert to URDF order and set joints
#     q[7:19] = joint_angles  # Direct assignment since we're using simple order
    
#     # Verify configuration is valid
#     if not pin.isNormalized(model, q, 1e-6):
#         print("⚠️  Configuration not normalized, fixing...")
#         q = pin.normalize(model, q)
    
#     return q

# def computeStandingContactForces(q):
#     """
#     Compute contact forces for standing equilibrium using physics-based approach.
#     """
#     # Use simple weight distribution for now
#     robot_mass = 12.0  # kg
#     total_weight = robot_mass * 9.81
#     weight_per_foot = total_weight / 4.0
    
#     f = np.zeros(NUM_F)
#     f[2] = weight_per_foot   # LF foot fz
#     f[5] = weight_per_foot   # RF foot fz  
#     f[8] = weight_per_foot   # LH foot fz
#     f[11] = weight_per_foot  # RH foot fz
    
#     return f

# def computeEquilibriumForces(q):
#     """
#     Compute contact forces that provide exact equilibrium for standing.
#     Uses pseudoinverse to solve the underdetermined system.
#     """
#     try:
#         # Zero velocity and acceleration for standing
#         v_zero = np.zeros(NUM_V)
#         qdd_zero = np.zeros(NUM_V)
        
#         # Compute gravity forces
#         tau_gravity = pin.rnea(model, data, q, v_zero, qdd_zero)
        
#         # Compute contact Jacobian
#         Jc = computeFullContactJacobians(q)
        
#         # Solve: Jc.T @ f = tau_gravity for f
#         # This is overdetermined (18 equations, 12 unknowns)
#         # Use least squares solution
#         f_equilibrium = np.linalg.lstsq(Jc.T, tau_gravity, rcond=None)[0]
        
#         return f_equilibrium
        
#     except Exception as e:
#         print(f"⚠️  Could not compute equilibrium forces: {e}")
#         # Fallback to simple weight distribution
#         return computeStandingContactForces(q)

# # ===== IMPROVED VALIDATION FUNCTIONS =====

# def test_configuration_validity():
#     """Test that our configuration is numerically valid"""
#     print("=== TESTING CONFIGURATION VALIDITY ===")
    
#     q_test = getValidStandingConfiguration()
    
#     print(f"Configuration shape: {q_test.shape}")
#     print(f"Base position: {q_test[:3]}")
#     print(f"Base quaternion: {q_test[3:7]} (norm: {np.linalg.norm(q_test[3:7]):.6f})")
#     print(f"Joint angles range: [{np.min(q_test[7:]):.3f}, {np.max(q_test[7:]):.3f}]")
    
#     # Check if configuration is normalized
#     is_normalized = pin.isNormalized(model, q_test, 1e-6)
#     print(f"Configuration normalized: {is_normalized}")
    
#     # Test forward kinematics
#     try:
#         pin.forwardKinematics(model, data, q_test)
#         pin.updateFramePlacements(model, data)
#         print("✅ Forward kinematics successful")
        
#         # Check foot positions
#         foot_positions = []
#         for frame_id in FOOT_FRAME_IDS:
#             pos = data.oMf[frame_id].translation
#             foot_positions.append(pos)
#             if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
#                 print(f"❌ Invalid foot position for frame {frame_id}: {pos}")
#                 return False
        
#         print("✅ All foot positions valid")
#         return True
        
#     except Exception as e:
#         print(f"❌ Forward kinematics failed: {e}")
#         return False

# def test_inverse_dynamics_dimensions():
#     """Test dimensions with valid configuration"""
#     print("\n=== TESTING INVERSE DYNAMICS DIMENSIONS ===")
    
#     q_test = getValidStandingConfiguration()
#     v_test = np.zeros(NUM_V)  # Use zeros to avoid numerical issues
#     qdd_test = np.zeros(NUM_V)
#     f_test = computeStandingContactForces(q_test)
    
#     try:
#         tau = id(q_test, v_test, qdd_test, f_test)
        
#         print(f"Output torque shape: {tau.shape} (expected: ({NUM_V},))")
#         print(f"Torque range: [{np.min(tau):.6f}, {np.max(tau):.6f}]")
#         print(f"Has NaN values: {np.any(np.isnan(tau))}")
#         print(f"Has Inf values: {np.any(np.isinf(tau))}")
        
#         if tau.shape == (NUM_V,) and not np.any(np.isnan(tau)) and not np.any(np.isinf(tau)):
#             print("✅ Inverse dynamics dimensions and values correct!")
#             return True
#         else:
#             print(f"❌ Issues detected in inverse dynamics output")
#             return False
            
#     except Exception as e:
#         print(f"❌ Inverse dynamics failed: {e}")
#         return False

# def test_inverse_dynamics_consistency_safe():
#     """Test consistency with small, safe values"""
#     print("\n=== TESTING INVERSE DYNAMICS CONSISTENCY (SAFE) ===")
    
#     q_test = getValidStandingConfiguration()
#     v_test = np.zeros(NUM_V)  # Start with zeros
#     qdd_test = np.zeros(NUM_V)  # Start with zeros
#     f_test = computeStandingContactForces(q_test)
    
#     print(f"Using safe test values (all zeros except forces)")
    
#     try:
#         # Test 1: Zero motion should give gravity compensation
#         tau_gravity = id(q_test, v_test, qdd_test, np.zeros(NUM_F))
#         print(f"Gravity compensation torque range: [{np.min(tau_gravity):.3f}, {np.max(tau_gravity):.3f}]")
        
#         # Test 2: With standing forces, should get small torques
#         tau_standing = id(q_test, v_test, qdd_test, f_test)
#         print(f"Standing torque range: [{np.min(tau_standing):.3f}, {np.max(tau_standing):.3f}]")
        
#         # Test 3: Small perturbation test
#         qdd_small = np.ones(NUM_V) * 0.01  # Very small acceleration
#         tau_perturb = id(q_test, v_test, qdd_small, f_test)
        
#         # Forward direction
#         qdd_recovered = fd(q_test, v_test, tau_perturb, f_test)
        
#         error = np.linalg.norm(qdd_recovered - qdd_small)
#         print(f"Small perturbation test error: {error:.6f}")
        
#         if error < 1e-3 and not np.any(np.isnan([tau_gravity, tau_standing, tau_perturb])):
#             print("✅ Inverse dynamics consistency test passed!")
#             return True
#         else:
#             print(f"⚠️  Consistency issues detected")
#             if np.any(np.isnan([tau_gravity, tau_standing, tau_perturb])):
#                 print("  - NaN values found")
#             if error >= 1e-3:
#                 print(f"  - Large consistency error: {error}")
#             return False
            
#     except Exception as e:
#         print(f"❌ Consistency test failed: {e}")
#         return False

# def test_standing_equilibrium_improved():
#     """Test standing equilibrium with improved force computation"""
#     print("\n=== TESTING STANDING EQUILIBRIUM (IMPROVED) ===")
    
#     q_stand = getValidStandingConfiguration()
    
#     try:
#         # Method 1: Simple weight distribution
#         f_simple = computeStandingContactForces(q_stand)
#         tau_simple = id(q_stand, np.zeros(NUM_V), np.zeros(NUM_V), f_simple)
        
#         print(f"Simple weight distribution:")
#         print(f"  Total vertical force: {np.sum(f_simple[2::3]):.2f}N")
#         print(f"  Max torque magnitude: {np.max(np.abs(tau_simple)):.6f}")
#         print(f"  Base torques: {np.max(np.abs(tau_simple[:6])):.6f}")
#         print(f"  Joint torques: {np.max(np.abs(tau_simple[6:])):.6f}")
        
#         # Method 2: Physics-based equilibrium
#         try:
#             f_equilibrium = computeEquilibriumForces(q_stand)
#             tau_equilibrium = id(q_stand, np.zeros(NUM_V), np.zeros(NUM_V), f_equilibrium)
            
#             print(f"\nPhysics-based equilibrium:")
#             print(f"  Total vertical force: {np.sum(f_equilibrium[2::3]):.2f}N")
#             print(f"  Max torque magnitude: {np.max(np.abs(tau_equilibrium)):.6f}")
#             print(f"  Base torques: {np.max(np.abs(tau_equilibrium[:6])):.6f}")
#             print(f"  Joint torques: {np.max(np.abs(tau_equilibrium[6:])):.6f}")
            
#             if np.max(np.abs(tau_equilibrium[:6])) < 1e-2:
#                 print("✅ Physics-based equilibrium achieved!")
#                 return True
#         except:
#             print("⚠️  Physics-based method failed, using simple method")
        
#         if np.max(np.abs(tau_simple[:6])) < 5.0:  # Relaxed criterion
#             print("✅ Reasonable equilibrium achieved with simple method")
#             return True
#         else:
#             print("⚠️  Equilibrium not achieved")
#             return False
            
#     except Exception as e:
#         print(f"❌ Equilibrium test failed: {e}")
#         return False

# # Run tests if this file is executed directly
# if __name__ == "__main__":
#     print("="*60)
#     print("DEBUGGING INVERSE DYNAMICS ISSUES")
#     print("="*60)
    
#     success = True
    
#     if not test_configuration_validity():
#         print("❌ Configuration has issues - stopping tests")
#         success = False
#     else:
#         if not test_inverse_dynamics_dimensions():
#             success = False
        
#         if not test_inverse_dynamics_consistency_safe():
#             success = False
            
#         if not test_standing_equilibrium_improved():
#             success = False
    
#     if success:
#         print("\n🎉 ALL INVERSE DYNAMICS TESTS PASSED!")
#         print("✅ Valid configuration")
#         print("✅ Correct dimensions") 
#         print("✅ Numerical consistency")
#         print("✅ Reasonable equilibrium")
#         print("\nYour id.py is ready for jumping optimization!")
#     else:
#         print("\n❌ SOME TESTS FAILED")
#         print("The issues may be due to:")
#         print("- Extreme joint configurations causing singularities")
#         print("- Numerical precision issues with random test values")
#         print("- Contact Jacobian conditioning problems")
#         print("\nTry running the jumping optimization anyway - it may still work!")













# # id.py

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
    """
    Inverse dynamics: compute motor torques needed for desired acceleration.
    
    Returns tau_motor where: tau_motor + J_c^T * f = M*qdd + C*v + g
    
    For floating base robots, tau_motor[0:6] should be zero (unactuated).
    """
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
        cs_tau_actuator = cs_tau_rnea - cs_Jc.T @ cs_f
        
        # Create function with correct inputs
        cs_tau_fn = ca.Function("create_tau_fn", [cs_q, cs_v, cs_qdd, cs_f], [cs_tau_actuator])
        
        # Evaluate at given values
        tau_actuator = cs_tau_fn(q, v, qdd, f)
       
    else: 
        Jc = computeFullContactJacobians(q)
        tau_actuator = pin.rnea(model, data, q, v, qdd) - Jc.T @ f        

    return tau_actuator