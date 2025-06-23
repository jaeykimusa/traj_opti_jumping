#!/usr/bin/env python3

import os
import numpy as np
import scipy.spatial.transform as st
import rerun as rr
import trimesh

import yourdfpy
from urdf_parser_py import urdf as urdf_parser

from mpac_logging import robot_zoo

import logging


class RobotLogger:
  def __init__(self, urdf: yourdfpy.URDF, base_link: str = None, prefix: str = "") -> None:
    self.urdf = urdf
    self.urdf_parser = urdf_parser.URDF.from_xml_string(self.urdf.write_xml_string())
    self.entity_to_transform = dict()
    self.meshes_logged = set()
    self.base_link = base_link if base_link else self.urdf.base_link
    self.prefix = prefix.rstrip("/")  # Remove trailing slash if present
    self.static_transforms_logged = False

  @classmethod
  def from_zoo(cls, robot_name: str, prefix: str = "") -> "RobotLogger":
    return cls(robot_zoo.get_urdf(robot_name), prefix=prefix)

  @property
  def joint_names(self):
    return list(self.urdf.actuated_joint_names)

  def link_entity_path(self, link: urdf_parser.Link) -> str:
    """Return the entity path for the URDF link."""
    link_names = self.urdf_parser.get_chain(self.urdf_parser.get_root(), link.name)[0::2]  # skip the joints
    path = "/".join(link_names)
    return f"{self.prefix}/{path}" if self.prefix else path

  def joint_entity_path(self, joint_name: str) -> str:
    root_name = self.urdf_parser.get_root()
    link_names = self.urdf_parser.get_chain(root_name, joint_name)[0::2]  # skip the joints
    path = "/".join(link_names)
    return f"{self.prefix}/{path}" if self.prefix else path

  def log_static_transforms(self):
    """Log the static geometry and initial state."""
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    # Log all links and their geometries (timeless)
    for link in self.urdf_parser.links:
      entity_path = self.link_entity_path(link)
      for i, visual in enumerate(link.visuals):
        self.log_visual(entity_path + f"/visual_{i}", visual)

  def log_state(self, *, joint_positions=None, base_position=None, base_orientation=None, logtime=None):
    """Log the current state (transforms only)."""
    # Apply base transform if provided
    rr.set_time_seconds("robot_time", logtime)

    if not self.static_transforms_logged:
      self.log_static_transforms()
      self.static_transforms_logged = True

    if base_position is not None and base_orientation is not None:
      # Create base transform
      rotation_matrix = st.Rotation.from_quat(base_orientation).as_matrix()
      base_path = f"{self.prefix}/{self.base_link}" if self.prefix else f"/{self.base_link}"
      rr.log(base_path, rr.Transform3D(translation=base_position, mat3x3=rotation_matrix))

    # Log joint transforms
    for joint_name in self.joint_names:
      joint = self.urdf.joint_map[joint_name]
      entity_path = self.joint_entity_path(joint.child)
      self.log_joint(entity_path, joint, joint_positions.get(joint_name, 0) if joint_positions else None)

  def log_joint(self, entity_path: str, joint, joint_angle=None) -> None:
    translation = rotation = None

    if joint.origin is not None:
      translation = joint.origin[:3, 3]
      rotation = joint.origin[:3, :3]

    # Apply joint angle if provided and joint is not fixed
    if joint_angle is not None and joint.type != "fixed" and joint.axis is not None:
      joint_rotation = st.Rotation.from_rotvec(joint_angle * joint.axis).as_matrix()
      if rotation is None:
        rotation = joint_rotation
      else:
        rotation = rotation @ joint_rotation

    self.entity_to_transform[entity_path] = (translation, rotation)
    rr.log(entity_path, rr.Transform3D(translation=translation, mat3x3=rotation))

  def log_visual(self, entity_path: str, visual) -> None:
    # Skip if we've already logged this mesh
    if entity_path in self.meshes_logged:
      return

    try:
      # default transform
      transform = np.eye(4)

      if visual.origin is not None:
        if visual.origin.xyz is not None:
          transform[:3, 3] = visual.origin.xyz
        if visual.origin.rpy is not None:
          transform[:3, :3] = st.Rotation.from_euler("xyz", visual.origin.rpy).as_matrix()

      if isinstance(visual.geometry, urdf_parser.Mesh):
        mesh_path = visual.geometry.filename
        logging.debug(f"Loading mesh from: {mesh_path}")

        if os.path.exists(mesh_path):
          logging.debug(f"File size: {os.path.getsize(mesh_path) / 1024 / 1024:.2f} MB")
          mesh_or_scene = trimesh.load_mesh(mesh_path)
        else:
          logging.warning(f"Mesh file not found: {mesh_path}")
          mesh_or_scene = trimesh.creation.box(extents=[0.1, 0.1, 0.1])

        if visual.geometry.scale is not None:
          transform[:3, :3] *= np.array(visual.geometry.scale)
      elif isinstance(visual.geometry, urdf_parser.Box):
        mesh_or_scene = trimesh.creation.box(extents=visual.geometry.size)
      elif isinstance(visual.geometry, urdf_parser.Cylinder):
        mesh_or_scene = trimesh.creation.cylinder(
          radius=visual.geometry.radius,
          height=visual.geometry.length,
        )
      elif isinstance(visual.geometry, urdf_parser.Sphere):
        mesh_or_scene = trimesh.creation.icosphere(
          radius=visual.geometry.radius,
        )
      else:
        logging.warning(f"Unsupported geometry type: {type(visual.geometry)}")
        mesh_or_scene = trimesh.creation.box(extents=[0.1, 0.1, 0.1])

      mesh_or_scene.apply_transform(transform)

      if isinstance(mesh_or_scene, trimesh.Scene):
        scene = mesh_or_scene
        for i, mesh in enumerate(scene.dump()):
          if visual.material is not None and visual.material.color is not None:
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.material.color.rgba
          self.log_trimesh(entity_path + f"/{i}", mesh)
          self.meshes_logged.add(entity_path + f"/{i}")
      else:
        mesh = mesh_or_scene
        if visual.material is not None and visual.material.color is not None and mesh_path.lower().endswith(".stl"):
          mesh.visual = trimesh.visual.ColorVisuals()
          mesh.visual.vertex_colors = visual.material.color.rgba
        self.log_trimesh(entity_path, mesh)
        self.meshes_logged.add(entity_path)

    except Exception as e:
      logging.error(f"Error loading visual for {entity_path}: {str(e)}")
      mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
      self.log_trimesh(entity_path, mesh)
      self.meshes_logged.add(entity_path)

  def log_trimesh(self, entity_path: str, mesh: trimesh.Trimesh) -> None:
    vertex_colors = albedo_texture = vertex_texcoords = None
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
      vertex_colors = mesh.visual.vertex_colors
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
      albedo_texture = mesh.visual.material.baseColorTexture
      if len(np.asarray(albedo_texture).shape) == 2:
        albedo_texture = np.stack([albedo_texture] * 3, axis=-1)
      vertex_texcoords = mesh.visual.uv
      if vertex_texcoords is not None:
        vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]

    rr.log(
      entity_path,
      rr.Mesh3D(
        vertex_positions=mesh.vertices,
        triangle_indices=mesh.faces,
        vertex_normals=mesh.vertex_normals,
        vertex_colors=vertex_colors,
        albedo_texture=albedo_texture,
        vertex_texcoords=vertex_texcoords,
      ),
    )
