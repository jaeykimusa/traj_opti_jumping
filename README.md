# traj_opti_jumping

Developing the first baseline of trajectory optimization for jumping controller. 

PROGRESS:

6/18/2025
- .
- TODO:
  - .

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
