import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import wandb
import math
from tqdm import tqdm

# === Config ===
GRID_SIZE      = 5
CELL_SIZE      = 0.3
NUM_ORIENT     = 72
NUM_FRAMES     = 32
NUM_ACTIONS    = GRID_SIZE * GRID_SIZE * NUM_ORIENT
HISTORY_LENGTH = 2 #4
NUM_EPOCHS     = 150
BATCH_SIZE     = 16 #32
LR             = 1e-4 #5e-4
GAMMA          = 0.99 #0.96
ENTROPY_COEF   = 0.001 #0.05
VALUE_COEF     = 0.5 #0.25
GRAD_CLIP      = 1.0
POS_WEIGHT     = 1.0
ORI_WEIGHT     = 2.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# === Feature extractor ===
class FeatureExtractor(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        #resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        if not fine_tune:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B, C, H, W)
        f = self.encoder(x)             # (B,512,1,1)
        return f.view(f.size(0), -1)    # (B,512)


# === Dataset ===
class PoseTrajectoryDataset(Dataset):
    def __init__(self, root_dir, transform=None, history=2):
        self.root_dir   = root_dir
        self.traj_dirs  = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.transform = transform or transforms.ToTensor()
        self.history   = history

    def __len__(self):
        return len(self.traj_dirs)

    def __getitem__(self, idx):
        traj_path = self.traj_dirs[idx]
        log_path  = os.path.join(traj_path, "log.csv")
        df        = pd.read_csv(log_path)

        frame_imgs, pose_data, action_ids = [], [], []
        for _, row in df.iterrows():
            img = Image.open(os.path.join(traj_path, row["img_path"])).convert("RGB")
            frame_imgs.append(self.transform(img))

            pose = torch.tensor([
                float(row["rel_x"]),
                float(row["rel_y"]),
                float(row["rel_theta"])
            ], dtype=torch.float32)
            pose_data.append(pose)

            action_ids.append(int(row.get("action_id", -1)))

        return {
            "frames":  torch.stack(frame_imgs),   # (T, C, H, W)
            "poses":   torch.stack(pose_data),    # (T, 3)
            "actions": torch.tensor(action_ids)   # (T,)
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


def pose_error(pred, gt, λ_pos=POS_WEIGHT, λ_ori=ORI_WEIGHT):
    dx, dy = pred[0] - gt[0], pred[1] - gt[1]
    pos_err = (dx*dx + dy*dy)**0.5
    ori_err = abs((pred[2] - gt[2] + 180) % 360 - 180)
    return λ_pos*pos_err**2 + λ_ori*(ori_err/180.0)**2


# === Actor‐Critic ===
class FeatureActorCriticNet(nn.Module):
    def __init__(self, input_dim=512, num_actions=NUM_ACTIONS):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


# === Training ===
def train_ppo_batched(
    model, feature_extractor, train_loader, val_loader,
    num_epochs, lr, gamma, entropy_coef, value_coef,
    grad_clip, device, save_dir
):
    wandb.watch(model, log="all", log_freq=100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val  = -float("inf")

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss, total_reward = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            frames = batch["frames"].to(device)   # (B,T,C,H,W)
            poses  = batch["poses"]               # CPU
            B, T, C, H, W = frames.shape

            # feature extraction
            imgs  = frames.view(B*T, C, H, W)
            feats = feature_extractor(imgs)       # (B*T,512)
            feats = feats.view(B, T, -1)          # (B,T,512)

            # build state vectors by concatenating f0 and ft when history=1
            if HISTORY_LENGTH == 1:
                f0     = feats[:, :1, :].expand(-1, T, -1)  # (B,T,512)
                states = torch.cat([f0, feats], dim=-1)      # (B,T,1024)
            else:
                f0     = feats[:, :1, :].expand(-1, T, -1)
                all_s  = []
                for t in range(T):
                    parts = [f0[:, t]]
                    for k in range(HISTORY_LENGTH-1, 0, -1):
                        parts.append(feats[:, t-k] if t-k>=0 else feats[:, 0])
                    parts.append(feats[:, t])
                    all_s.append(torch.cat(parts, dim=-1))
                states = torch.stack(all_s, dim=1)  # (B,T,(H+1)*512)

            # policy & value
            flat       = states.view(B*T, -1)
            logits, val = model(flat)              # (B*T,A), (B*T,1)
            dist       = Categorical(logits=logits)
            actions    = dist.sample()              # (B*T,)
            logp       = dist.log_prob(actions)     # (B*T,)
            entropy    = dist.entropy().mean()

            # rewards
            flat_actions = actions.cpu().numpy()
            flat_poses   = poses.view(-1, 3).numpy()
            rewards = torch.tensor(
                #[-pose_error(action_to_pose(int(a)), p)
                [math.exp(-pose_error(action_to_pose(int(a)), p))
                 for a,p in zip(flat_actions, flat_poses)],
                device=device
            )

            # discounted returns
            returns = torch.zeros_like(rewards)
            R = 0.0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + gamma*R
                returns[i] = R

            vals       = val.view(-1)
            advantages = returns - vals.detach()

            # losses
            policy_loss = -(logp*advantages).mean()
            value_loss  = (returns-vals).pow(2).mean()
            loss        = policy_loss + value_coef*value_loss - entropy_coef*entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)	# What makes this PPO
            optimizer.step()

            # logging
            total_loss   += loss.item()
            batch_reward = rewards.sum().item()/(B*T)
            total_reward += batch_reward
            wandb.log({
                "batch/loss": loss.item(),
                "batch/reward": batch_reward,
                "epoch": epoch
            })
            pbar.set_postfix(loss=loss.item(), reward=batch_reward)

        # epoch logging
        wandb.log({
            "epoch/train_loss":   total_loss/len(train_loader),
            "epoch/train_reward": total_reward/len(train_loader),
            "epoch": epoch
        })

        # validation
        if val_loader is not None:
            model.eval()
            val_rewards = []
            with torch.no_grad():
                for batch in val_loader:
                    frames = batch["frames"].to(device)
                    poses  = batch["poses"]
                    B, T, C, H, W = frames.shape

                    imgs  = frames.view(B*T, C, H, W)
                    feats = feature_extractor(imgs).view(B, T, -1)

                    if HISTORY_LENGTH == 1:
                        f0     = feats[:, :1, :].expand(-1, T, -1)
                        states = torch.cat([f0, feats], dim=-1)
                    else:
                        f0     = feats[:, :1, :].expand(-1, T, -1)
                        all_s  = []
                        for t in range(T):
                            parts = [f0[:, t]]
                            for k in range(HISTORY_LENGTH-1, 0, -1):
                                parts.append(feats[:, t-k] if t-k>=0 else feats[:, 0])
                            parts.append(feats[:, t])
                            all_s.append(torch.cat(parts, dim=-1))
                        states = torch.stack(all_s, dim=1)

                    logits, _    = model(states.view(B*T, -1))
                    actions_val = Categorical(logits=logits).sample().cpu().numpy()
                    flat_poses  = poses.view(-1, 3).numpy()
                    for a,p in zip(actions_val, flat_poses):
                        #val_rewards.append(-pose_error(action_to_pose(int(a)), p))
                        val_rewards.append(math.exp(-pose_error(action_to_pose(int(a)), p)))

            avg_val = sum(val_rewards)/len(val_rewards)
            print(f"[Validation] Avg Reward: {avg_val:.4f}")
            wandb.log({"epoch/val_reward": avg_val, "epoch": epoch})
            if avg_val > best_val:
                best_val = avg_val
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f"best_{epoch}.pt"))


def main():
    CHECKPOINT_PATH = None  # or "/path/to/your/checkpoint.pt"
    DATASET_PATH    = "/data/students/earl/ai322/rldr/trajectories"
    SAVE_DIR        = "/data/students/earl/ai322/rldr/checkpoints"
    RUN_DIR         = os.path.join(SAVE_DIR, f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(RUN_DIR, exist_ok=True)

    wandb.init(
        project="ppo_train_concath_trials",
        dir=RUN_DIR,
        config={
            "history_length": HISTORY_LENGTH,
            "epochs":       NUM_EPOCHS,
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "gamma":        GAMMA,
            "entropy_coef": ENTROPY_COEF,
            "value_coef":   VALUE_COEF
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.GaussianBlur(3, (0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    feature_extractor = FeatureExtractor(fine_tune=False).to(device)

    # use new dataset class (no CNN inside)
    ds = PoseTrajectoryDataset(DATASET_PATH, transform=val_t, history=HISTORY_LENGTH)
    train_len = int(0.7 * len(ds))
    val_len   = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    val_ds.dataset.transform = val_t

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # match the concatenation: 512 * (HISTORY_LENGTH + 1)
    input_dim = 512 * (HISTORY_LENGTH + 1)
    model     = FeatureActorCriticNet(input_dim=input_dim).to(device)

    if CHECKPOINT_PATH:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt)

    train_ppo_batched(
        model, feature_extractor,
        train_loader, val_loader,
        NUM_EPOCHS, LR, GAMMA,
        ENTROPY_COEF, VALUE_COEF,
        GRAD_CLIP, device, RUN_DIR
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
