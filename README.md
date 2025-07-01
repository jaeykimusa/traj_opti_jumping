# traj_opti_jumping

Developing the first baseline of trajectory optimization for jumping controller. 

PROGRESS:

7/1/2025
- .
- TODO:
  - 

6/30/2025
- Abstract--
- TODO:
  - continue using srbd to compute reference trajectory and implement contact timing optimization in that process.
  - determine the best variable frequency maneuver.
  - whole-body controller for landing phase.

6/19/2025
- Computing ground reation forces works well, verified and stored in dynamics.py.
- Added bunch of print methods for the output. 
- TODO:
  - Use qp and id to make it stand solely and that can be iterated in the core controller.
  - Start jumping.

6/18/2025
- computed the ground reaction forces at each contact point. + moments. However, there is a big force in x direction which reduce the amount of normal forces on z direction. 
- Got rerun work on mac. Just added "absl-py" as one of dependecies in mpac_logging -> pyproject.toml. And just pip3 install -e . no need to install rerun nor rerun-sdk externally. 
- TODO:
  - Add dynamics constraints to prohibit any infeasible contact force in x direction.
  - Compute qp solver to make it stand dynamically.

6/17/2025:
- I think i did make the default standing position on the ground where all the ee at z = 0, and it will calculate floating base level using qp.
- TODO:
  - Add dynamics constraints to make it stand.
  - Determine the ground reaction forces at each contact point.
  - Maybe I can control for it to go up and down by adjusting its legs.
  - Start trajectory optimization code in 3d.

6/16/2025: 
- uploaded robot model for go2 via pinocchio.
- setted visualization via mpac_logging and rerun.
- ran some basic vis using general state vectors.
- TODO:
  - make it stand using dynamics constraints.
  - recall the ground reaction forces to blance, and i will go from there. to make it jump...

6/15/2025:
- tba

6/14/2025:
- tba

6/13/2025:
- tba
