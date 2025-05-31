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
using an Advantage Actor-Critic (A2C) algorithm with Proximal Policy Optimization (PPO).
The action space is quantized using $5\times5\times 72$ discrete actions representing
25 Cartesian positions and 72 binned orientation.

An alternative approach, which takes advantage of the full continuous action space, is
the Deep Deterministic Policy Gradient (DDPG) algorithm, which is also an A2C with PPO,
but without the action quantization.

## Training

The main training scripts are `ppo_training.py` and `ddpg_training.py` for A2C and DDPG, 
respectively. Hyperparameter tuning was implemented using the `ppo_training_hypertune.py`
and `ddpg_training_hypertune.py` scripts. 

## Inference

The inference script `infer_utils.py` require ROS2 to execute on the robot. The notebooks
`map_inference_a2c.ipynb` and `map_inference_ddpg.ipynb` posted here simply reuse the existing 
dataset to demonstrate the pose prediction performance of models `best_72.pt` for A2C and 
`best_ddpg_2.pt` for DDPG. These models offer substantial opportunities for improvement.
the RL model. 
