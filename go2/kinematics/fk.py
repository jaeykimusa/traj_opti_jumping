# # fk.py

# from go2.robot.robot import *
# from go2.robot.morphology import *
# from go2.utils.math_utils import *

# # def fk(q):
# #     if isinstance(q, (ca.SX, ca.MX)):
# #         cs_x = ca.SX.sym("x", 15, 1)
# #         cs_q = ca.SX.sym("q", NUM_Q, 1)
# #         pinocchio.casadi.forwardKinematics(ad_model, ad_data, cs_q)
# #         bodyPos = data.oMf[BASE_FRAME].translation        
# #         eePos = []
# #         for frameId in EE_FRAME_IDS:
# #             ee_pos = data.oMf[frameId].translation
# #             eePos.append(ee_pos)
# #         cs_x = ca.vertcat(
# #             bodyPos,
# #             *eePos
# #         )
# #         cs_x_fn = ca.Function("x_fn", [cs_q], [cs_x])
# #         x = cs_x_fn(q)
# #     else:
# #         x = []
# #         pin.forwardKinematics(model, data, q)
# #         bodyPosition = data.oMf[Frame.BASE_LINK].translation
# #         x[:3] = bodyPosition
# #         for frameId in EE_FRAME_IDS:
# #             EEPosition = data.oMf[frameId].translation
# #             x = np.hstack((x, EEPosition))
# #     return x

# # ===== CORRECTED fk() FUNCTION =====

# def fk(q):
#     """
#     Forward kinematics: compute body and foot positions.
    
#     Args:
#         q: Configuration vector (NUM_Q = 19 for FreeFlyer model)
        
#     Returns:
#         x: Positions vector (15,) = [body_x, body_y, body_z, foot1_x, foot1_y, foot1_z, ..., foot4_x, foot4_y, foot4_z]
#            Order: [body_pos, LF_foot, RF_foot, LH_foot, RH_foot]
#     """
    
#     # Define foot frame names (same as in fd.py and id.py)
#     FOOT_FRAME_NAMES = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    
#     # Get foot frame IDs
#     FOOT_FRAME_IDS = []
#     for frame_name in FOOT_FRAME_NAMES:
#         if model.existFrame(frame_name):
#             frame_id = model.getFrameId(frame_name)
#             FOOT_FRAME_IDS.append(frame_id)
#         else:
#             raise ValueError(f"Could not find foot frame: {frame_name}")
    
#     if isinstance(q, (ca.SX, ca.MX)):
#         # CasADi symbolic mode
#         cs_q = ca.SX.sym("q", NUM_Q, 1)
        
#         # Forward kinematics
#         pinocchio.casadi.forwardKinematics(ad_model, ad_data, cs_q)
#         pinocchio.casadi.updateFramePlacements(ad_model, ad_data, cs_q)
        
#         # FIXED: Get body position directly from configuration
#         # For FreeFlyer model, base position is q[0:3]
#         bodyPos = cs_q[:3]  # Direct from configuration vector
        
#         # Get foot positions using correct frame IDs
#         eePos = []
#         for frameId in FOOT_FRAME_IDS:
#             # FIXED: Use ad_data (CasADi data) instead of data
#             ee_pos = ad_data.oMf[frameId].translation
#             eePos.append(ee_pos)
        
#         # Concatenate body and foot positions
#         cs_x = ca.vertcat(bodyPos, *eePos)  # Shape: (15,)
        
#         # Create function
#         cs_x_fn = ca.Function("fk_function", [cs_q], [cs_x])
#         x = cs_x_fn(q)
        
#     else:
#         # Numerical mode
#         pin.forwardKinematics(model, data, q)
#         pin.updateFramePlacements(model, data)
        
#         # FIXED: Get body position directly from configuration
#         # For FreeFlyer model, base position is q[0:3]
#         bodyPosition = q[:3]  # Direct from configuration vector
        
#         # Initialize result array
#         x = np.zeros(15)
#         x[:3] = bodyPosition
        
#         # Get foot positions using correct frame IDs
#         for i, frameId in enumerate(FOOT_FRAME_IDS):
#             EEPosition = data.oMf[frameId].translation
#             start_idx = 3 + i * 3  # Body (3) + previous feet (i*3)
#             x[start_idx:start_idx+3] = EEPosition
    
#     return x

# # ===== ALTERNATIVE: MORE ROBUST VERSION =====

# def fk_robust(q):
#     """
#     More robust forward kinematics with better error handling.
#     """
#     # Define foot frame names consistently
#     FOOT_FRAME_NAMES = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    
#     # Get frame IDs with error checking
#     try:
#         FOOT_FRAME_IDS = []
#         for frame_name in FOOT_FRAME_NAMES:
#             if model.existFrame(frame_name):
#                 frame_id = model.getFrameId(frame_name)
#                 FOOT_FRAME_IDS.append(frame_id)
#             else:
#                 print(f"⚠️  Frame {frame_name} not found, checking alternatives...")
#                 # Try alternative names
#                 alt_names = [f"{frame_name}_joint", f"{frame_name}_link", 
#                            frame_name.replace("_FOOT", ""), frame_name.lower()]
#                 found = False
#                 for alt_name in alt_names:
#                     if model.existFrame(alt_name):
#                         frame_id = model.getFrameId(alt_name)
#                         FOOT_FRAME_IDS.append(frame_id)
#                         print(f"✅ Using alternative: {alt_name}")
#                         found = True
#                         break
#                 if not found:
#                     raise ValueError(f"Could not find any frame for: {frame_name}")
        
#         print(f"✅ FK using frames: {[(name, id) for name, id in zip(FOOT_FRAME_NAMES, FOOT_FRAME_IDS)]}")
        
#     except Exception as e:
#         print(f"❌ Frame setup failed: {e}")
#         print("Available frames:")
#         for i in range(min(10, model.nframes)):
#             print(f"  {i}: {model.frames[i].name}")
#         raise
    
#     if isinstance(q, (ca.SX, ca.MX)):
#         # CasADi symbolic mode
#         cs_q = ca.SX.sym("q", NUM_Q, 1)
        
#         # Forward kinematics
#         pinocchio.casadi.forwardKinematics(ad_model, ad_data, cs_q)
#         pinocchio.casadi.updateFramePlacements(ad_model, ad_data, cs_q)
        
#         # Body position from configuration
#         bodyPos = cs_q[:3]
        
#         # Foot positions
#         foot_positions = []
#         for frameId in FOOT_FRAME_IDS:
#             ee_pos = ad_data.oMf[frameId].translation
#             foot_positions.append(ee_pos)
        
#         # Stack all positions
#         result = ca.vertcat(bodyPos, *foot_positions)
        
#         # Create function
#         fk_fn = ca.Function("fk_robust", [cs_q], [result])
#         return fk_fn(q)
        
#     else:
#         # Numerical mode
#         pin.forwardKinematics(model, data, q)
#         pin.updateFramePlacements(model, data)
        
#         # Body position from configuration
#         body_pos = q[:3]
        
#         # Foot positions
#         foot_positions = []
#         for frameId in FOOT_FRAME_IDS:
#             foot_pos = data.oMf[frameId].translation
#             foot_positions.append(foot_pos)
        
#         # Concatenate all
#         result = np.concatenate([body_pos] + foot_positions)
#         return result

# # ===== VALIDATION FUNCTION =====

# def test_fk_function():
#     """Test the corrected FK function"""
#     print("=== TESTING CORRECTED FK FUNCTION ===")
    
#     # Test with valid configuration
#     q_test = pin.neutral(model)
#     q_test[6] = 1.0  # Set quaternion w component
#     q_test[2] = 0.3  # Set height
    
#     print(f"Test configuration shape: {q_test.shape}")
#     print(f"Base position from config: {q_test[:3]}")
    
#     try:
#         # Test FK function
#         x_result = fk(q_test)
        
#         print(f"FK result shape: {x_result.shape} (expected: (15,))")
#         print(f"Body position from FK: {x_result[:3]}")
#         print(f"LF foot position: {x_result[3:6]}")
#         print(f"RF foot position: {x_result[6:9]}")
#         print(f"LH foot position: {x_result[9:12]}")
#         print(f"RH foot position: {x_result[12:15]}")
        
#         # Check for NaN/Inf values
#         has_nan = np.any(np.isnan(x_result))
#         has_inf = np.any(np.isinf(x_result))
        
#         print(f"Has NaN values: {has_nan}")
#         print(f"Has Inf values: {has_inf}")
        
#         if x_result.shape == (15,) and not has_nan and not has_inf:
#             print("✅ FK function working correctly!")
            
#             # Check if body position matches configuration
#             body_error = np.linalg.norm(x_result[:3] - q_test[:3])
#             print(f"Body position consistency: {body_error:.6f}")
            
#             if body_error < 1e-6:
#                 print("✅ Body position consistency verified!")
#             else:
#                 print("⚠️  Body position mismatch")
                
#             return True
#         else:
#             print("❌ FK function has issues")
#             return False
            
#     except Exception as e:
#         print(f"❌ FK test failed: {e}")
#         return False

# # ===== USAGE EXAMPLE =====

# def example_usage():
#     """Show how to use the corrected FK function"""
#     print("\n=== FK FUNCTION USAGE EXAMPLE ===")
    
#     # Get a standing configuration
#     try:
#         from go2.robot.robot import getDefaultStandStateFullOptimization
#         q_stand = getDefaultStandStateFullOptimization(model, data)
#     except:
#         q_stand = pin.neutral(model)
#         q_stand[6] = 1.0  # quaternion w
#         q_stand[2] = 0.3  # height
    
#     # Compute forward kinematics
#     positions = fk(q_stand)
    
#     print(f"Standing configuration FK:")
#     print(f"  Body: [{positions[0]:.3f}, {positions[1]:.3f}, {positions[2]:.3f}]")
#     print(f"  LF:   [{positions[3]:.3f}, {positions[4]:.3f}, {positions[5]:.3f}]")
#     print(f"  RF:   [{positions[6]:.3f}, {positions[7]:.3f}, {positions[8]:.3f}]")
#     print(f"  LH:   [{positions[9]:.3f}, {positions[10]:.3f}, {positions[11]:.3f}]")
#     print(f"  RH:   [{positions[12]:.3f}, {positions[13]:.3f}, {positions[14]:.3f}]")

# # # Run tests if executed directly
# # if __name__ == "__main__":
# #     print("="*60)
# #     print("TESTING CORRECTED FORWARD KINEMATICS")
# #     print("="*60)
    
# #     if test_fk_function():
# #         example_usage()
# #         print("\n🎉 FK FUNCTION READY FOR OPTIMIZATION!")
# #     else:
# #         print("\n❌ FK FUNCTION NEEDS MORE FIXES")
# #         print("Try using fk_robust() instead of fk()")













# fk.py

from go2.robot.robot import *
from go2.robot.morphology import *
from go2.utils.math_utils import *

def fk(q):
    if isinstance(q, (ca.SX, ca.MX)):
        cs_x = ca.SX.sym("x", 15, 1)
        cs_q = ca.SX.sym("q", NUM_Q, 1)
        pinocchio.casadi.forwardKinematics(ad_model, ad_data, cs_q)
        bodyPos = data.oMf[BASE_FRAME].translation        
        eePos = []
        for frameId in EE_FRAME_IDS:
            ee_pos = data.oMf[frameId].translation
            eePos.append(ee_pos)
        cs_x = ca.vertcat(
            bodyPos,
            *eePos
        )
        cs_x_fn = ca.Function("x_fn", [cs_q], [cs_x])
        x = cs_x_fn(q)
    else:
        x = []
        pin.forwardKinematics(model, data, q)
        bodyPosition = data.oMf[BASE_FRAME].translation
        x[:3] = bodyPosition
        for frameId in EE_FRAME_IDS:
            EEPosition = data.oMf[frameId].translation
            x = np.hstack((x, EEPosition))
    return x

