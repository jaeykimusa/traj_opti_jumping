#!/usr/bin/env python3

import rerun as rr
import rerun.blueprint as rrb


def rerun_initialize(name: str, spawn: bool = False) -> None:
  rr.init(name, spawn=spawn)

  # Background and grid
  blueprint = rrb.Blueprint(
    rrb.Spatial3DView(
      origin="/",
      name="3D Scene",
      background=[65, 0, 255],
      line_grid=rrb.archetypes.LineGrid3D(
        visible=True,
        spacing=0.1,
        stroke_width=0.1,
        color=[220, 255, 0],
      ),
    ),
    collapse_panels=False,
  )
  rr.send_blueprint(blueprint)

  # Origin axes
  rr.log(
    "origin_axes",
    rr.Arrows3D(
      origins=[[0, 0, 0]] * 3,
      vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
      labels=["X", "Y", "Z"],
      radii=0.025,
    ),
    static=True,
  )


def rerun_store(name: str) -> None:
  rr.save(name)


__all__ = ["rerun_initialize", "rerun_store"]
