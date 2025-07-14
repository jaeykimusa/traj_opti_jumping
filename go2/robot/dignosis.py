import pinocchio as pin
import numpy as np
from pathlib import Path

print("=== GO2 ROBOT MODEL DIAGNOSIS AND FIX ===")

# ===== STEP 1: ANALYZE CURRENT MODEL =====
print("\n1. Analyzing current robot model...")

# Load your current model
pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
urdf_filename = pin_model_dir / "go2.urdf"

# Your current setup
joint_model = pin.JointModelComposite(2)
joint_model.addJoint(pin.JointModelTranslation())
joint_model.addJoint(pin.JointModelSphericalZYX())

model_current = pin.buildModelFromUrdf(str(urdf_filename), joint_model)
data_current = model_current.createData()

print(f"Current model dimensions:")
print(f"  nq (config dim): {model_current.nq}")
print(f"  nv (velocity dim): {model_current.nv}")
print(f"  njoints: {model_current.njoints}")

print(f"\nCurrent joint names and types:")
for i, joint in enumerate(model_current.joints):
    print(f"  Joint {i}: {model_current.names[i]} - {type(joint).__name__}")

# ===== STEP 2: STANDARD FREEFLYAR MODEL =====
print("\n2. Creating standard FreeFlyer model for comparison...")

model_standard = pin.buildModelFromUrdf(str(urdf_filename), pin.JointModelFreeFlyer())
data_standard = model_standard.createData()

print(f"Standard FreeFlyer model dimensions:")
print(f"  nq (config dim): {model_standard.nq}")
print(f"  nv (velocity dim): {model_standard.nv}")
print(f"  njoints: {model_standard.njoints}")

print(f"\nStandard joint names and types:")
for i, joint in enumerate(model_standard.joints):
    print(f"  Joint {i}: {model_standard.names[i]} - {type(joint).__name__}")

# ===== STEP 3: JOINT ORDER ANALYSIS =====
print("\n3. Analyzing joint ordering...")

print(f"\nActuated joints in URDF order:")
urdf_joint_names = []
for i in range(1, model_standard.njoints):  # Skip universe joint
    joint_name = model_standard.names[i]
    if joint_name != "root_joint":  # Skip floating base
        urdf_joint_names.append(joint_name)
        print(f"  {len(urdf_joint_names)-1}: {joint_name}")

# Expected order for quadruped (common convention)
expected_order = [
    "LF_HAA", "LF_HFE", "LF_KFE",  # Left Front: Hip_AA, Hip_FE, Knee_FE
    "RF_HAA", "RF_HFE", "RF_KFE",  # Right Front
    "LH_HAA", "LH_HFE", "LH_KFE",  # Left Hind
    "RH_HAA", "RH_HFE", "RH_KFE"   # Right Hind
]

print(f"\nExpected joint order:")
for i, name in enumerate(expected_order):
    print(f"  {i}: {name}")

# Check if ordering matches
order_matches = urdf_joint_names == expected_order
print(f"\nJoint order matches expected: {order_matches}")

if not order_matches:
    print("❌ Joint ordering mismatch detected!")
    print("Actual order:", urdf_joint_names)
    print("Expected order:", expected_order)

# ===== STEP 4: CONFIGURATION SPACE ANALYSIS =====
print("\n4. Configuration space analysis...")

# Test random configurations
q_current = pin.randomConfiguration(model_current)
q_standard = pin.randomConfiguration(model_standard)

print(f"Current model config vector (nq={len(q_current)}):")
print(f"  Base translation: {q_current[:3]}")
print(f"  Base rotation (ZYX Euler): {q_current[3:6]}")
print(f"  Joint angles: {q_current[6:]}")

print(f"\nStandard model config vector (nq={len(q_standard)}):")
print(f"  Base translation: {q_standard[:3]}")
print(f"  Base quaternion: {q_standard[3:7]}")
print(f"  Joint angles: {q_standard[7:]}")

# ===== STEP 5: FORWARD KINEMATICS TEST =====
print("\n5. Testing forward kinematics...")

# Set both models to same standing pose
q_test_current = np.zeros(model_current.nq)
q_test_current[2] = 0.3  # Height
q_test_current[8] = 0.8   # Joint example
q_test_current[10] = -1.6

q_test_standard = np.zeros(model_standard.nq)
q_test_standard[2] = 0.3  # Height  
q_test_standard[6] = 1.0  # Quaternion w
q_test_standard[9] = 0.8  # Joint example (offset by +1 due to quaternion)
q_test_standard[11] = -1.6

try:
    pin.forwardKinematics(model_current, data_current, q_test_current)
    pin.updateFramePlacements(model_current, data_current)
    
    pin.forwardKinematics(model_standard, data_standard, q_test_standard)
    pin.updateFramePlacements(model_standard, data_standard)
    
    print("✅ Forward kinematics successful for both models")
    
    # Compare base positions
    base_pos_current = data_current.oMi[1].translation  # Root joint
    base_pos_standard = data_standard.oMi[1].translation
    
    print(f"Base position current: {base_pos_current.T}")
    print(f"Base position standard: {base_pos_standard.T}")
    
except Exception as e:
    print(f"❌ Forward kinematics failed: {e}")

# ===== STEP 6: RECOMMENDED FIXES =====
print(f"\n" + "="*60)
print("6. RECOMMENDED FIXES")
print("="*60)

print(f"\n🔧 ISSUE 1: Configuration Space Mismatch")
print(f"   Current: 6 DOF floating base (translation + ZYX Euler)")
print(f"   Standard: 6 DOF floating base (translation + quaternion)")
print(f"   → This causes dimension mismatches in optimization")

print(f"\n🔧 ISSUE 2: Optimization Assumes Standard FreeFlyer")
print(f"   Your optimization code assumes:")
print(f"   - q[0:3] = base position")
print(f"   - q[3:6] = base orientation (but you use Euler, not quaternion)")
print(f"   - q[6:18] = joint angles")

print(f"\n💡 SOLUTION OPTIONS:")

print(f"\n   Option A: Switch to Standard FreeFlyer (RECOMMENDED)")
print(f"   ✅ Change robot.py to use JointModelFreeFlyer()")
print(f"   ✅ Update optimization for quaternion representation")
print(f"   ✅ Most compatible with Pinocchio ecosystem")

print(f"\n   Option B: Fix Current Model")
print(f"   ⚠️  Keep current joint model but fix optimization")
print(f"   ⚠️  More complex, less standard")

print(f"\n📋 IMMEDIATE ACTION ITEMS:")

print(f"\n1. Check joint name ordering:")
if order_matches:
    print(f"   ✅ Joint names are in correct order")
else:
    print(f"   ❌ Joint names need reordering in optimization")
    print(f"   → Update your joint indexing in constraints")

print(f"\n2. Fix floating base representation:")
print(f"   → Update robot.py to use JointModelFreeFlyer()")
print(f"   → Update optimization constraints for quaternion base")

print(f"\n3. Verify dimensions:")
print(f"   Current NUM_Q = {model_current.nq}, NUM_Q should match model.nq")
print(f"   Standard NUM_Q = {model_standard.nq}")

# ===== STEP 7: GENERATE FIXED ROBOT.PY =====
print(f"\n7. Generating fixed robot.py...")

fixed_robot_code = f'''# Fixed robot.py for Go2 jumping optimization
from pathlib import Path
from sys import argv
from enum import Enum, auto

import pinocchio as pin
from go2.robot.morphology import *
import numpy as np
import casadi as ca
import pinocchio.casadi

# Model directory
pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
urdf_filename = (
    pin_model_dir / "go2.urdf"
    if len(argv) < 2
    else argv[1]
)

# FIXED: Use standard FreeFlyer instead of custom composite joint
# This ensures compatibility with optimization code
model = pin.buildModelFromUrdf(str(urdf_filename), pin.JointModelFreeFlyer())
robot = pin.RobotWrapper(model)
data = model.createData()

ad_model = pinocchio.casadi.Model(model)
ad_data = ad_model.createData()

# Print model info for verification
print(f"Go2 Robot Model Loaded:")
print(f"  Configuration dimension (nq): {{model.nq}}")
print(f"  Velocity dimension (nv): {{model.nv}}")
print(f"  Number of joints: {{model.njoints}}")

# Verify joint order
actuated_joints = []
for i in range(1, model.njoints):
    if model.names[i] != "root_joint":
        actuated_joints.append(model.names[i])

expected_joints = [
    "LF_HAA", "LF_HFE", "LF_KFE",  # Left Front
    "RF_HAA", "RF_HFE", "RF_KFE",  # Right Front  
    "LH_HAA", "LH_HFE", "LH_KFE",  # Left Hind
    "RH_HAA", "RH_HFE", "RH_KFE"   # Right Hind
]

print(f"\\nJoint ordering verification:")
for i, (actual, expected) in enumerate(zip(actuated_joints, expected_joints)):
    status = "✅" if actual == expected else "❌"
    print(f"  {{i}}: {{actual}} (expected: {{expected}}) {{status}}")

# Define corrected constants for optimization
NUM_Q = model.nq  # Should be 19 (7 for floating base + 12 for joints)
NUM_V = model.nv  # Should be 18 (6 for floating base + 12 for joints)  
NUM_F = 12        # 4 feet × 3 force components
NUM_JOINTS = 12   # 4 legs × 3 joints per leg

print(f"\\nConstants for optimization:")
print(f"  NUM_Q = {{NUM_Q}}")
print(f"  NUM_V = {{NUM_V}}")
print(f"  NUM_F = {{NUM_F}}")
print(f"  NUM_JOINTS = {{NUM_JOINTS}}")
'''

print("Fixed robot.py code generated!")
print("This code uses standard FreeFlyer joint which should fix the jumping issues.")

print(f"\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
print("\n🎯 NEXT STEPS:")
print("1. Replace your robot.py with the fixed version above")
print("2. Update NUM_Q in morphology.py to match model.nq (likely 19)")
print("3. Verify optimization uses q[7:19] for joint angles (not q[6:18])")
print("4. Test the jumping optimization again")
print("\nThe main issue is your floating base uses Euler angles (6 DOF)")
print("but optimization assumes quaternion representation (7 DOF)!")