from pathlib import Path
from sys import argv
from enum import Enum, auto

import pinocchio as pin
from go2.robot.morphology import *
import numpy as np

import casadi as ca
import pinocchio.casadi

# This path refers to pin source code but you can define your own directory here.
pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = (
    pin_model_dir / "go2.urdf"
    if len(argv) < 2
    else argv[1]
)
joint_model = pin.JointModelComposite(2)
joint_model.addJoint(pin.JointModelTranslation())
joint_model.addJoint(pin.JointModelSphericalZYX())
# symbolic term
model = pin.buildModelFromUrdf(urdf_filename, joint_model)
robot = pin.RobotWrapper(model)
data = model.createData()

ad_model = pinocchio.casadi.Model(model)
ad_data = ad_model.createData()

# from pathlib import Path
# from sys import argv
# from enum import Enum, auto

# import pinocchio as pin
# from go2.robot.morphology import *
# import numpy as np
# import casadi as ca
# import pinocchio.casadi

# # Model directory
# pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
# urdf_filename = (
#     pin_model_dir / "go2.urdf"
#     if len(argv) < 2
#     else argv[1]
# )

# # CRITICAL FIX: Use standard FreeFlyer for quaternion-based floating base
# model = pin.buildModelFromUrdf(str(urdf_filename), pin.JointModelFreeFlyer())
# robot = pin.RobotWrapper(model)
# data = model.createData()

# ad_model = pinocchio.casadi.Model(model)
# ad_data = ad_model.createData()

# print(f"✅ Fixed Go2 Robot Model Loaded:")
# print(f"   nq = {model.nq} (was 18, now 19 - includes quaternion)")
# print(f"   nv = {model.nv} (still 18 - velocity space)")

# # ===== 2. FIXED MORPHOLOGY.PY - Update your constants =====

# # Updated constants to match actual model dimensions
# NUM_Q = 19  # FIXED: was 18, now 19 (includes quaternion)
# NUM_V = 18  # Velocity space dimension
# NUM_F = 12  # Contact forces (4 feet × 3 components)

# print(f"✅ Updated constants: NUM_Q={NUM_Q}, NUM_V={NUM_V}, NUM_F={NUM_F}")

# # ===== 3. JOINT ORDER MAPPING - Handle the URDF joint ordering =====

# # Your URDF actual order (from diagnosis)
# URDF_JOINT_ORDER = [
#     "LF_HAA", "LF_HFE", "LF_KFE",  # Left Front  (indices 0-2)
#     "LH_HAA", "LH_HFE", "LH_KFE",  # Left Hind   (indices 3-5)  
#     "RF_HAA", "RF_HFE", "RF_KFE",  # Right Front (indices 6-8)
#     "RH_HAA", "RH_HFE", "RH_KFE"   # Right Hind  (indices 9-11)
# ]

# # Standard quadruped order (what your optimization expects)
# STANDARD_JOINT_ORDER = [
#     "LF_HAA", "LF_HFE", "LF_KFE",  # Left Front  (indices 0-2)
#     "RF_HAA", "RF_HFE", "RF_KFE",  # Right Front (indices 3-5)
#     "LH_HAA", "LH_HFE", "LH_KFE",  # Left Hind   (indices 6-8)  
#     "RH_HAA", "RH_HFE", "RH_KFE"   # Right Hind  (indices 9-11)
# ]

# # Create mapping from URDF order to standard order
# JOINT_MAPPING = np.array([
#     0, 1, 2,   # LF stays: 0→0, 1→1, 2→2
#     6, 7, 8,   # RF moves: 6→3, 7→4, 8→5  
#     3, 4, 5,   # LH moves: 3→6, 4→7, 5→8
#     9, 10, 11  # RH stays: 9→9, 10→10, 11→11
# ])

# def reorder_joints_urdf_to_standard(q_urdf):
#     """Convert joint angles from URDF order to standard order"""
#     q_standard = q_urdf.copy()
#     if len(q_urdf) == 12:  # Just joint angles
#         q_standard = q_urdf[JOINT_MAPPING]
#     elif len(q_urdf) == 19:  # Full configuration
#         q_standard[7:19] = q_urdf[7:19][JOINT_MAPPING]  # Reorder joints only
#     return q_standard

# def reorder_joints_standard_to_urdf(q_standard):
#     """Convert joint angles from standard order to URDF order"""
#     q_urdf = q_standard.copy()
#     if len(q_standard) == 12:  # Just joint angles
#         q_urdf[JOINT_MAPPING] = q_standard
#     elif len(q_standard) == 19:  # Full configuration
#         q_urdf[7:19][JOINT_MAPPING] = q_standard[7:19]
#     return q_urdf

# print(f"✅ Joint reordering functions created")

# # ===== 4. FIXED INITIAL CONFIGURATION FUNCTION =====

# def getDefaultStandStateFullOptimization(model, data):
#     """Get corrected standing configuration for FreeFlyer model"""
#     q = pin.neutral(model)  # Start with neutral config
    
#     # Set base position
#     q[0] = 0.0    # x
#     q[1] = 0.0    # y  
#     q[2] = 0.297  # z (height)
    
#     # Set base orientation (quaternion [x, y, z, w])
#     q[3] = 0.0  # qx
#     q[4] = 0.0  # qy
#     q[5] = 0.0  # qz  
#     q[6] = 1.0  # qw (normalized)
    
#     # Set joint angles in STANDARD order (what optimization expects)
#     standard_joint_angles = np.array([
#         0.0,  0.8, -1.5,   # LF: HAA, HFE, KFE
#         0.0,  0.8, -1.5,   # RF: HAA, HFE, KFE
#         0.0,  0.8, -1.5,   # LH: HAA, HFE, KFE  
#         0.0,  0.8, -1.5    # RH: HAA, HFE, KFE
#     ])
    
#     # Convert to URDF order before setting in q
#     urdf_joint_angles = standard_joint_angles[JOINT_MAPPING]
#     q[7:19] = urdf_joint_angles  # Joints start at index 7 in FreeFlyer
    
#     return q

# def computeStandingContactForces(q):
#     """Compute standing contact forces for the corrected model"""
#     robot_mass = 12.0  # kg
#     weight_per_foot = robot_mass * 9.81 / 4.0
    
#     # Forces in standard order: [fx, fy, fz] × 4 feet
#     f = np.zeros(12)
#     f[2] = weight_per_foot   # LF foot fz
#     f[5] = weight_per_foot   # RF foot fz  
#     f[8] = weight_per_foot   # LH foot fz
#     f[11] = weight_per_foot  # RH foot fz
    
#     return f

# print(f"✅ Fixed initial configuration functions created")