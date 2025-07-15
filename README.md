# traj_opti_jumping

## Install
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba create -n p310 python=3.10 pinocchio=3.7.0
micromamba activate p310
pip3 install -r ./requirements.txt
```

## Run
```bash
TRAJOPT_LOG_LEVEL=info python3 run.py
```