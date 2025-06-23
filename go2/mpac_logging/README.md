# mpac_logging 

## Showcase


![output2](https://github.com/user-attachments/assets/2d2686af-d172-4585-a409-30bed890bb94)
![output3](https://github.com/user-attachments/assets/bb4aea85-e036-43d1-805c-7d5761582ebd)


## Zoo


There are many robots in the zoo and we rely on the amazing [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) project that does the model downloading and loading on our behalf.

```python3
from mpac_logging.robot_zoo import all_robots
from mpac_logging.robot_zoo import get_urdf

print(all_robots())

```

```python3
from mpac_logging.robot_logger import RobotLogger
go2_logger = RobotLogger.from_zoo("go2_description", prefix="go2")
g1_logger = RobotLogger.from_zoo("g1_description", prefix="g1")
```

## Install dependencies
```bash
git clone git@github.com:Hier-Lab/mpac_logging.git && cd mpac_logging
pip3 install -e .
```

## TODOs

- [ ] Switch completely to pinocchio, use nothing else but pinocchio
