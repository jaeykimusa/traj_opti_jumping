#!/usr/bin/env python3

import argparse
import itertools
import numpy as np
import time
import scipy.spatial.transform as st
from mpac_logging.rerun.robot_logger import RobotLogger
from mpac_logging.rerun.utils import rerun_initialize, rerun_store

import logging


def generate_circular_motion(t: float, center: np.ndarray, radius: float, frequency: float):
  """Generate circular motion pattern."""
  x = center[0] + radius * np.cos(2 * np.pi * frequency * t)
  y = center[1] + radius * np.sin(2 * np.pi * frequency * t)
  z = center[2]
  position = np.array([x, y, z])

  # Make robot face direction of motion
  dx = -radius * 2 * np.pi * frequency * np.sin(2 * np.pi * frequency * t)
  dy = radius * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
  yaw = np.arctan2(dy, dx)
  orientation = st.Rotation.from_euler("xyz", [0, 0, yaw]).as_quat()

  return position, orientation


def generate_figure_eight(t: float, center: np.ndarray, scale: float, frequency: float):
  """Generate figure-8 pattern."""
  x = center[0] + scale * np.sin(2 * np.pi * frequency * t)
  y = center[1] + scale * np.sin(4 * np.pi * frequency * t)
  z = center[2] + 0.1 * np.sin(2 * np.pi * frequency * 2 * t)  # Add slight bounce
  position = np.array([x, y, z])

  # Calculate direction of motion for orientation
  dx = scale * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
  dy = scale * 4 * np.pi * frequency * np.cos(4 * np.pi * frequency * t)
  yaw = np.arctan2(dy, dx)
  orientation = st.Rotation.from_euler("xyz", [0, 0, yaw]).as_quat()

  return position, orientation


def main(args):
  logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] | %(message)s", level=args.log_level.upper()
  )
  # Initialize rerun
  rerun_initialize("multi_robot_example", spawn=False)

  # Create robot loggers for different robots
  go2_logger = RobotLogger.from_zoo(args.robot1, prefix="prefix_" + args.robot1)
  g1_logger = RobotLogger.from_zoo(args.robot2, prefix="prefix_" + args.robot2)

  # Log initial states
  current_time = time.time()

  try:
    t = 0
    for i in itertools.count():
      current_time = time.time()

      # GO2 robot: Circular motion
      go2_pos, go2_ori = generate_circular_motion(t, center=np.array([0, 0, 0.5]), radius=1.5, frequency=0.2)

      # G1 robot: Figure-8 pattern
      g1_pos, g1_ori = generate_figure_eight(t, center=np.array([0, 0, 0.5]), scale=2.0, frequency=0.15)

      # Generate joint motions
      go2_joints = {name: 0.3 * np.sin(2 * np.pi * 0.5 * t) for name in go2_logger.joint_names}
      g1_joints = {name: 0.4 * np.sin(2 * np.pi * 0.7 * t) for name in g1_logger.joint_names}

      # Log states for all robots
      go2_logger.log_state(
        logtime=current_time, base_position=go2_pos, base_orientation=go2_ori, joint_positions=go2_joints
      )

      g1_logger.log_state(
        logtime=current_time, base_position=g1_pos, base_orientation=g1_ori, joint_positions=g1_joints
      )

      if i % 50 == 0:
        logging.info("Running...")

      t += 0.05
      time.sleep(0.05)

  except KeyboardInterrupt:
    print("\nSaving recording...")
    rerun_store("multi_robot.rrd")
    print("Run `rerun multi_robot.rrd` to view the recording.")
    print("Alternatively, go to rerun.io/viewer and drag the generated file to the viewer.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--log_level", type=str, default="INFO", help="The log level.")
  parser.add_argument("--robot1", type=str, default="go2_description", help="The name of the first robot to log.")
  parser.add_argument("--robot2", type=str, default="g1_description", help="The name of the second robot to log.")
  args = parser.parse_args()
  main(args)
