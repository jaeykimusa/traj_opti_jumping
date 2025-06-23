#!/usr/bin/env python3

import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robot_descriptions import DESCRIPTIONS


def all_robots() -> list[str]:
  return [name for (name, value) in DESCRIPTIONS.items() if value.has_urdf]


def get_urdf(robot_name: str) -> yourdfpy.URDF:
  return load_robot_description(robot_name)


__all__ = ["get_urdf", "all_robots"]
