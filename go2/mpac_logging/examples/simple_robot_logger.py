#!/usr/bin/env python3

import argparse
import itertools
import numpy as np
import time
from mpac_logging import robot_zoo
from mpac_logging.rerun.robot_logger import RobotLogger
from mpac_logging.rerun.utils import rerun_initialize, rerun_store

import logging


def main(args):
  logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] | %(message)s", level=args.log_level.upper()
  )
  rerun_initialize("simple_robot_example", spawn=False)

  # Create a robot logger for the GO2 robot

  print(robot_zoo.all_robots())
  robot_logger = RobotLogger.from_zoo(args.robot_name)

  # Log the initial state (static geometry)
  current_time = time.time()

  try:
    t = 0
    for i in itertools.count():
      current_time = time.time()

      # Generate a simple circular motion for the base
      radius = 1.0
      x = radius * np.cos(t)
      y = radius * np.sin(t)
      z = 0.5  # Fixed height

      # Create a simple orientation that makes the robot face the direction of motion
      yaw = t  # Robot faces the direction of motion
      orientation = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])  # Simple yaw rotation as quaternion

      # Create some simple joint motion - just sine waves
      joint_positions = {joint_name: 0.5 * np.sin(2 * t) for joint_name in robot_logger.joint_names}

      # Log the new state
      robot_logger.log_state(
        logtime=current_time, base_position=[x, y, z], base_orientation=orientation, joint_positions=joint_positions
      )

      if i % 50 == 0:
        logging.info("Running...")

      t += 0.05
      time.sleep(0.05)

  except KeyboardInterrupt:
    logging.info("Saving recording...")
    rerun_store("simple_robot.rrd")
    logging.info("Run `rerun simple_robot.rrd` to view the recording.")
    logging.info("Alternatively, go to rerun.io/viewer and drag the generated file to the viewer.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--robot_name", type=str, default="go2_description", help="The name of the robot to log.")
  parser.add_argument("--log_level", type=str, default="INFO", help="The log level.")
  args = parser.parse_args()
  main(args)
