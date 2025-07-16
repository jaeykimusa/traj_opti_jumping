# import pinocchio as pin
# from pinocchio.visualize import MeshcatVisualizer
# import numpy as np
# import os

# urdf_path = "ur3.urdf"
# model =  pin.buildModelFromUrdf(urdf_path)
# data = model.createData()

# #viz = MeshcatVisualizer(model)
# #viz.initViewer()
# #viz.loadViewerModel()


# q = np.zeros(model.nq)
# v = np.zeros(model.nv)
# tau = np.zeros(model.nv)

# pin.forwardKinematics(model, data, q)
# pin.computeJointJacobians(model, data, q)
# pin.framesForwardKinematics(model, data, q)


import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

# Initialize MeshCat visualizer
vis = meshcat.Visualizer()

# Load UR3 URDF (replace with your URDF file path)
urdf_path = "assets/models/ur3.urdf"  # Replace with actual path
try:
    with open(urdf_path, 'r') as f:
        urdf_string = f.read()
except FileNotFoundError:
    print(f"Error: URDF file not found at {urdf_path}")
    exit()

# Add UR3 robot to the scene
vis["robot"].set_object(g.Mesh(urdf_string))

# Example: Move the robot
joint_angles = np.array([0.5, -1.0, 0.8, 0.2, -0.7, 0.0])  # Example joint angles

# You'll need a function to convert joint angles to transforms
# (This depends on your robot description and kinematics)
# For this example, I am assuming that the urdf is already setup to receive joint angles

# Assuming your URDF uses joint names like "shoulder_pan_joint", "shoulder_lift_joint", etc.
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

for i, name in enumerate(joint_names):
    vis["robot"][name].set_transform(tf.rotation_matrix(joint_angles[i], [0, 0, 1]))

# Open the visualizer in your browser
vis.open()
