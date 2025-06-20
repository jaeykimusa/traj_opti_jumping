from pathlib import Path
from sys import argv
from enum import Enum, auto

import pinocchio as pin
from .morphology import *

# This path refers to pin source code but you can define your own directory here.
pin_model_dir = Path(__file__).parent.parent / "robot/go2_description"
 
# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = (
    pin_model_dir / "model/go2/go2.urdf"
    if len(argv) < 2
    else argv[1]
)
 
# Load the urdf model
model = pin.buildModelFromUrdf(urdf_filename, pin.JointModelFreeFlyer())
geom_model = pin.buildGeomFromUrdf(model, urdf_filename, pin.GeometryType.COLLISION)
robot = pin.RobotWrapper(model)
data = model.createData()
geom_data = geom_model.createData()
