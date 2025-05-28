"""
Hyperparameter tuning script for PPO training of a feature actor-critic network

1. Checks for available GPU memory and launches training jobs on them (MIN_FREE_MEM_GB).
2. Considers WANDB rate limits (MAX_CONCURRENT_RUNS).

"""

import os
import datetime
import math
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import wandb
from tqdm import tqdm
import GPUtil
import time

# === Global constants ===
GRID_SIZE      = 5
CELL_SIZE      = 0.3
NUM_ORIENT     = 72
NUM_FRAMES     = 32
NUM_ACTIONS    = GRID_SIZE * GRID_SIZE * NUM_ORIENT
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# === Feature extractor ===
class FeatureExtractor(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        if not fine_tune:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.encoder(x)             # (B,512,1,1)
        return f.view(f.size(0), -1)    # (B,512)

# === Dataset ===
class PoseTrajectoryDataset(Dataset):
    def __init__(self, root_dir, transform=None, history=1):
        self.root_dir   = root_dir
        self.traj_dirs  = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.transform = transform or transforms.ToTensor()
        self.history   = history

    def __len__(self):
        return len(self.traj_dirs)

    def __getitem__(self, idx):
        traj = self.traj_dirs[idx]
        df   = pd.read_csv(os.path.join(self.root_dir, traj, "log.csv"))
        frames, poses, acts = [], [], []
        for _, r in df.iterrows():
            img = Image.open(os.path.join(self.root_dir, traj, r["img_path"])).convert("RGB")
            frames.append(self.transform(img))
            poses.append(torch.tensor(
                [r["rel_x"], r["rel_y"], r["rel_theta"]],
                dtype=torch.float32
            ))
            acts.append(int(r.get("action_id", -1)))
        return {
            "frames":  torch.stack(frames),       # (T, C, H, W)
            "poses":   torch.stack(poses),        # (T, 3)
            "actions": torch.tensor(acts, dtype=torch.long)  # (T,)
        }

# === Utils ===
def action_to_pose(action_id):
    ang = action_id % NUM_ORIENT
    idx = action_id // NUM_ORIENT
    row, col = divmod(idx, GRID_SIZE)
    x = (col - GRID_SIZE//2)*CELL_SIZE
    y = (row - GRID_SIZE//2)*CELL_SIZE
    θ = ang*(360.0/NUM_ORIENT)
    return [x, y, θ]

def pose_error(pred, gt, λ_pos, λ_ori):
    dx, dy = pred[0]-gt[0], pred[1]-gt[1]
    pos_err = math.hypot(dx, dy)
    ori_err = abs((pred[2]-gt[2]+180)%360 - 180)
    return λ_pos*pos_err**2 + λ_ori*(ori_err/180.0)**2

# === Actor‐Critic ===
class FeatureActorCriticNet(nn.Module):
    def __init__(self, input_dim, num_actions, dropout_p=0.2):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(512, 256),       nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(256, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(512, 256),       nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# === Training loop ===
def train_ppo(hparams, gpu_id):
    # assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # unpack hyperparameters
    lr           = hparams["lr"]
    gamma        = hparams["gamma"]
    entropy_coef = hparams["entropy_coef"]
    value_coef   = hparams["value_coef"]
    grad_clip    = hparams["grad_clip"]
    history      = hparams["history"]
    run_name     = f"lr{lr}-g{gamma}-e{entropy_coef}-v{value_coef}-c{grad_clip}-h{history}"

    # initialize wandb
    wandb.init(
        project="ppo_concath_train_hypertune2",
        name=run_name,
        config=hparams
    )

    # transforms & dataset
    train_t = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    val_t = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    ds = PoseTrajectoryDataset(
        root_dir=hparams["data_path"],
        transform=val_t, history=history
    )
    train_len = int(0.7*len(ds))
    val_len   = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    val_ds.dataset.transform = val_t

    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
                              #shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=hparams["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # model & extractor
    feat_ext = FeatureExtractor(fine_tune=False).to(device)
    input_dim = 512 * (history + 1)
    model    = FeatureActorCriticNet(input_dim, NUM_ACTIONS).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(feat_ext.parameters()),
        lr=lr
    )

    best_val = -float("inf")
    for epoch in range(1, hparams["epochs"]+1):
        model.train()
        for batch in tqdm(train_loader, desc=f"{run_name} Epoch{epoch}"):
            frames = batch["frames"].to(device)  # (B,T,C,H,W)
            poses  = batch["poses"]
            B, T, C, H, W = frames.shape

            imgs  = frames.view(B*T, C, H, W)
            feats = feat_ext(imgs).view(B, T, -1)  # (B, T, 512)
            # build state vectors with dynamic history
            if history == 1:
                # initial + current
                f0     = feats[:, :1, :].expand(-1, T, -1)  # (B, T, 512)
                states = torch.cat([f0, feats], dim=-1)      # (B, T, 1024)
            else:
                all_s = []
                for t in range(T):
                    # gather 'history' past frames (pad with first frame)
                    hist = []
                    for i in range(history):
                        idx = max(0, t - i)
                        hist.append(feats[:, idx, :])         # (B, 512)
                    hist.reverse()                            # oldest→newest
                    # concat initial + history frames
                    state = torch.cat([feats[:, 0, :]] + hist, dim=-1)  # (B, (history+1)*512)
                    all_s.append(state)
                # stack per time step
                states = torch.stack(all_s, dim=1)  # (B, T, (history+1)*512)
            # flatten for actor‐critic
            states = states.view(B*T, -1)           # (B*T, input_dim)

            logits, val = model(states)
            dist   = Categorical(logits=logits)
            acts   = dist.sample()
            logp   = dist.log_prob(acts)
            entropy= dist.entropy().mean()

            # reward
            flat_acts = acts.cpu().numpy()
            flat_pose = poses.view(-1,3).numpy()
            rewards = torch.tensor([
                math.exp(-pose_error(action_to_pose(int(a)), p,
                                     hparams["pos_weight"],
                                     hparams["ori_weight"]))
                for a,p in zip(flat_acts, flat_pose)
            ], device=device)

            # returns & advantages
            returns = torch.zeros_like(rewards)
            R = 0.0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + gamma*R
                returns[i] = R
            vals = val.view(-1)
            advs = returns - vals.detach()

            # losses
            policy_loss = -(logp*advs).mean()
            value_loss  = (returns-vals).pow(2).mean()
            loss        = policy_loss + value_coef*value_loss - entropy_coef*entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            wandb.log({
                "train/loss": loss.item(),
                "train/reward": rewards.mean().item(),
                "train/epoch": epoch
            })

        # validation
        model.eval()
        val_rewards = []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                poses  = batch["poses"]
                B, T, C, H, W = frames.shape
                imgs  = frames.view(B*T, C, H, W)
                feats = feat_ext(imgs).view(B, T, -1)    # (B,T,512)
                if history == 1:
                    f0     = feats[:, :1, :].expand(-1, T, -1)  # (B,T,512)
                    states = torch.cat([f0, feats], dim=-1)      # (B,T,1024)
                else:
                    # build per-time-step state with initial frame + last `history` frames
                    all_s = []
                    for t in range(T):
                        hist = []
                        for i in range(history):
                            idx = max(0, t - i)
                            hist.append(feats[:, idx, :])          # (B,512)
                        hist.reverse()                             # oldest → newest
                        state = torch.cat([feats[:, 0, :]] + hist, dim=-1)
                        all_s.append(state)                      # list of (B,512*(history+1))
                    states = torch.stack(all_s, dim=1)           # (B,T,512*(history+1))
                states = states.view(B*T, -1) 
                logits, _ = model(states)
                acts = Categorical(logits=logits).sample().cpu().numpy()
                for a,p in zip(acts, poses.view(-1,3).numpy()):
                    val_rewards.append(math.exp(-pose_error(
                        action_to_pose(int(a)), p,
                        hparams["pos_weight"], hparams["ori_weight"]
                    )))
        avg_val = sum(val_rewards)/len(val_rewards)    
        wandb.log({"val/val_reward": avg_val, "epoch": epoch})
        if avg_val > best_val:
            best_val = avg_val
            torch.save(model.state_dict(),
                       os.path.join(hparams["save_dir"],
                                    f"{run_name}_best.pth"))

    wandb.finish()

# === Hyperparameter grid & launcher ===
def launcher():
    mp.set_start_method('spawn', force=True)
    # define hyperparameter options
    lr_list           = [5e-4]#[1e-4, 5e-4, 1e-3]
    gamma_list        = [0.99]#[0.95, 0.99]
    entropy_list      = [0.001, 0.005, 0.01]
    value_list        = [0.25, 0.5, 1.0]
    clip_list         = [0.5, 1.0, 2.0]
    history_list      = [3, 2, 1, 4, 5]  
    
    # fixed fields
    common = {
        "data_path":  "/data/students/earl/ai322/rldr/trajectories",
        "save_dir":   "/data/students/earl/ai322/rldr/checkpoints",
        "epochs":     150,
        "batch_size": 16,
        "pos_weight": 1.0,
        "ori_weight": 2.0
    }
    # create a top‐level RUN_DIR with timestamp under save_dir
    root_save_dir = os.path.join(
        common["save_dir"],
        f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )
    os.makedirs(root_save_dir, exist_ok=True)

    # build all combinations
    grid = list(itertools.product(
        lr_list, gamma_list, entropy_list, value_list, clip_list, history_list
    ))

    # prepare a queue of jobs
    jobs = []
    for lr, γ, e_coef, v_coef, g_clip, history in grid:
        run_name = f"lr{lr}-g{γ}-e{e_coef}-v{v_coef}-c{g_clip}-h{history}"
        run_dir  = os.path.join(root_save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        hparams = dict(common,
                        lr=lr,
                        gamma=γ,
                        entropy_coef=e_coef,
                        value_coef=v_coef,
                        grad_clip=g_clip,
                        history=history,
                        save_dir=run_dir)
        jobs.append(hparams)

    # minimum free memory per job in GB
    MIN_FREE_MEM_GB = 10
    # maximum concurrent wandb runs
    MAX_CONCURRENT_RUNS = 9


    # track active (process, gpu_id)
    active = []
    

    while jobs or active:
        # query all GPUs and select those with enough free memory
        free_gpus = {str(g.id)
                     for g in GPUtil.getGPUs()
                     if g.memoryFree/1024 >= MIN_FREE_MEM_GB}

        # compute candidates: free and not currently used by active procs
        busy_gpus = {gpu for _, gpu in active}
        candidates = list(free_gpus - busy_gpus)

        # launch jobs while we have candidates and pending jobs
        while candidates and jobs and len(active) < MAX_CONCURRENT_RUNS:
            hparams = jobs.pop(0)
            gpu     = candidates.pop(0)
            p = mp.Process(target=train_ppo, args=(hparams, gpu))
            p.start()
            active.append((p, gpu))

        # wait and reap finished jobs
        time.sleep(5)
        for p, gpu in active.copy():
            if not p.is_alive():
                p.join()
                active.remove((p, gpu))

    print("All hyperparameter runs completed.")

if __name__ == "__main__":
    launcher()