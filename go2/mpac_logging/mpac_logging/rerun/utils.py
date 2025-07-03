#!/usr/bin/env python3

import rerun as rr
import rerun.blueprint as rrb


def rerun_initialize(name: str, spawn: bool = False) -> None:
    rr.init(name, spawn=spawn)

    # Log a simulated ground plane (green)
    rr.log("world/ground", rr.Mesh3D(
        vertex_positions=[
            [-100, -100, 0],  # corner 1
            [100, -100, 0],   # corner 2
            [100, 100, 0],    # corner 3
            [-100, 100, 0],   # corner 4
        ],
        triangle_indices=[
            [0, 1, 2],
            [0, 2, 3],
        ],
        vertex_colors=[[0, 0, 0]] * 4,  # forest green
    ))

    # Background and grid view
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            name="3D Scene",
            background=[240, 240, 240],  # light sky blue
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True,
                spacing=0.25,
                stroke_width=1,
                color=[28, 84, 158],
            ),
        ),
        collapse_panels=False,
    )

    rr.send_blueprint(blueprint)

  # Origin axes
  # rr.log(
  #   "origin_axes",
  #   rr.Arrows3D(
  #     origins=[[0, 0, 0]] * 3,
  #     vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  #     colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
  #     labels=["X", "Y", "Z"],
  #     radii=0.025,
  #   ),
  #   static=True,
  # )


def rerun_store(name: str) -> None:
  rr.save(name)


__all__ = ["rerun_initialize", "rerun_store"]
