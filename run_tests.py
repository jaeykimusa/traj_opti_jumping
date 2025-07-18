#!/usr/bin/env python3

# from trajopt_logging import get_logger
# import pinocchio as pin
# from robot_descriptions.loaders.pinocchio import load_robot_description

# logger = get_logger("trajopt", "go2")

# logger.info("Loading pinocchio model for go2")
# joint_model = pin.JointModelComposite(2)
# joint_model.addJoint(pin.JointModelTranslation())
# joint_model.addJoint(pin.JointModelSphericalZYX())
# desc = load_robot_description("go2_description", joint_model)
# model, data = desc.model, desc.data

# frame_names = [frame.name for frame in model.frames]
# for name in frame_names:
#     print(name)
# exit()
# for joint_id, name in enumerate(model.names):
#     joint = model.joints[joint_id]
#     print(f"Joint ID: {joint_id}, Name: {name}, Joint Type: {type(joint).__name__}")

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(8, 8), squeeze=False)  # Force axs to be a 2D array
axs = axs.flatten()  # Now this works

x = np.array([1, 2, 3])
y = np.array([1, 4, 9])

axs[0].plot(x, y, label="x_d")

axs[0].set_title("desired vs optimal position")
axs[0].legend()
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

plt.tight_layout()
plt.show()
exit()