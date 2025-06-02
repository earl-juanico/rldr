# HOWTO:
This repository contains the full trajectory dataset used for training and
inference. The following sections discuss details about some details about
the RL method and training procedure. You may skip these sections to go
straight to the inference with visualizations.


## Dataset

The dataset was gathered using a Turtlebot3 Burger operated with ROS2 Humble.
The data collection was automated with the code `data_collector_levy.py`, which
implements a constrained Levy walk to move the robot randomly while capturing
snapshots with an onboard RaspberryPi camera.

**Structure:**
```
trajectories/
├── traj_000/
│   ├── final_pose.txt
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   ├── ...
│   ├── log.csv
│   └── start_pose.txt
├── traj_001/
│   └── ...
├── ...
└── traj_1278/
```

The `heatmap_data_collection.ipynb` visualizes the distribution of the robot poses in
a 5x5 grid with the start pose at the center of this grid. The robot's orientation
is expressed as angle $\theta$ in degrees with $0^{\circ}$ pointing north.

## RL Method

The pose prediction problem is solved by teaching a model with reinforcement learning
using Deep Deterministic Policy Gradient (DDPG) algorithm that assumes a continuous
action space for pose estimation.

A discretized model Advantage Actor-Critic (A2C) model with proximal policy optimization
is used for hyperparameter tuning before the full DDPG is trained. 


## Training

The main training script is the `ddpg_training.py` for DDPG. 
Hyperparameter tuning was implemented using the `ppo_training_hypertune.py`.
 

## Inference
The inference script `infer_utils.py` require ROS2 to execute on the robot. The notebook
`map_inference_ddpg.ipynb` posted here is the notebook that contains the details of 
performing inferences that simply reuses the existing 
dataset to demonstrate the pose prediction performance of checkpoints found in the 
directory `infer_trajectory`.

If you do not have physical access to the robot, you may skip the first cell of this 
notebook.
