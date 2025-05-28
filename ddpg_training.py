import os
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import wandb
from tqdm import tqdm
import copy

# === Config ===
NUM_FRAMES     = 32
HISTORY_LENGTH = 2
NUM_EPOCHS     = 150
BATCH_SIZE     = 16
LR_ACTOR       = 1e-4
LR_CRITIC      = 1e-4
GAMMA          = 0.95
TAU            = 0.005  # target smoothing
GRAD_CLIP      = 1.0
POS_WEIGHT     = 1.0
ORI_WEIGHT     = 2.0
REPLAY_SIZE    = 100_000
REPLAY_INIT    = 1000
BATCH_REPLAY   = 128
MAX_ACTION     = torch.tensor([1.0, 1.0, 1.0])  # x, y in meters, theta normalized to [-1, 1]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        f = self.encoder(x)
        return f.view(f.size(0), -1)

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

        frame_imgs, pose_data = [], []
        for _, row in df.iterrows():
            img = Image.open(os.path.join(traj_path, row["img_path"])).convert("RGB")
            frame_imgs.append(self.transform(img))
            # Normalize rel_theta from degrees to [-1, 1]
            rel_theta_norm = ((float(row["rel_theta"]) + 180) % 360 - 180) / 180.0
            pose = torch.tensor([
                float(row["rel_x"]),
                float(row["rel_y"]),
                rel_theta_norm
            ], dtype=torch.float32)
            pose_data.append(pose)

        return {
            "frames":  torch.stack(frame_imgs),   # (T, C, H, W)
            "poses":   torch.stack(pose_data),    # (T, 3)
        }

# === Utils ===
def pose_error(pred, gt, λ_pos=POS_WEIGHT, λ_ori=ORI_WEIGHT):
    dx, dy = pred[0] - gt[0], pred[1] - gt[1]
    pos_err = (dx*dx + dy*dy)**0.5
    # Orientation error in normalized [-1, 1] space, wrap to [-1, 1]
    ori_diff = (pred[2] - gt[2] + 1) % 2 - 1
    ori_err = abs(ori_diff)
    # Scale orientation error back to degrees for interpretability
    return λ_pos * pos_err**2 + λ_ori * (ori_err)**2

# === Ornstein-Uhlenbeck Noise ===
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, max_size=REPLAY_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []

    def add(self, state, action, reward, next_state, done):
        if self.size < self.max_size:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.next_state.append(next_state)
            self.done.append(done)
            self.size += 1
        else:
            idx = self.ptr
            self.state[idx] = state
            self.action[idx] = action
            self.reward[idx] = reward
            self.next_state[idx] = next_state
            self.done[idx] = done
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(np.array([self.state[i] for i in idxs])).to(device),
            torch.FloatTensor(np.array([self.action[i] for i in idxs])).to(device),
            torch.FloatTensor(np.array([self.reward[i] for i in idxs])).unsqueeze(1).to(device),
            torch.FloatTensor(np.array([self.next_state[i] for i in idxs])).to(device),
            torch.FloatTensor(np.array([self.done[i] for i in idxs])).unsqueeze(1).to(device)
        )

# === DDPG Actor & Critic ===
class Actor(nn.Module):
    def __init__(self, input_dim, max_action):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
        )
        self.x_head = nn.Linear(256, 1)
        self.y_head = nn.Linear(256, 1)
        self.theta_head = nn.Linear(256, 1)
        self.max_action = max_action

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        # Use sigmoid for x and y, then scale to [-max_action, max_action]
        x_pred = (torch.sigmoid(self.x_head(h)) * 2 - 1) * self.max_action[0]
        y_pred = (torch.sigmoid(self.y_head(h)) * 2 - 1) * self.max_action[1]
        # Use tanh for theta as before
        theta_pred = torch.tanh(self.theta_head(h)) * self.max_action[2]
        return torch.cat([x_pred, y_pred, theta_pred], dim=1)

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim + action_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q = self.trunk(x)
        return q

# === Training ===
def train_ddpg(
    actor, actor_target, critic, critic_target, feature_extractor,
    train_loader, val_loader,
    num_epochs, lr_actor, lr_critic, gamma, tau, grad_clip, device, save_dir
):
    wandb.watch(actor, log="all", log_freq=100)
    wandb.watch(critic, log="all", log_freq=100)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
    best_val  = float("inf")

    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    ou_noise = OUNoise(action_dim=3)

    # === Pre-fill replay buffer with random policy ===
    print("Pre-filling replay buffer...")
    with torch.no_grad():
        for batch in tqdm(train_loader, total=REPLAY_INIT//BATCH_SIZE+1, desc="Replay prefill"):
            frames = batch["frames"].to(device)
            poses  = batch["poses"].to(device)
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
            flat_states = states.view(B*T, -1).cpu().numpy()
            flat_targets = poses.view(B*T, 3).cpu().numpy()
            for i in range(flat_states.shape[0]-1):
                s = flat_states[i]
                a = flat_targets[i] + np.random.randn(3)*0.05  # random action
                # Reward shaping: exp(-pose_error)
                r = np.exp(-pose_error(a, flat_targets[i]))
                #r = -pose_error(a, flat_targets[i])
                s2 = flat_states[i+1]
                d = 0.0 if i < flat_states.shape[0]-2 else 1.0
                replay_buffer.add(s, a, r, s2, d)
                if replay_buffer.size >= REPLAY_INIT:
                    break
            if replay_buffer.size >= REPLAY_INIT:
                break

    for epoch in range(1, num_epochs+1):
        actor.train()
        critic.train()
        total_actor_loss, total_critic_loss, total_reward = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            frames = batch["frames"].to(device)
            poses  = batch["poses"].to(device)
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
            flat_states = states.view(B*T, -1)
            flat_targets = poses.view(B*T, 3)

            # === DDPG interaction: select action with exploration ===
            for i in range(flat_states.size(0)-1):
                s = flat_states[i].cpu().numpy()
                a = actor(flat_states[i].unsqueeze(0)).detach().cpu().numpy().flatten()
                a += ou_noise.sample() * MAX_ACTION.cpu().numpy() * 0.1  # exploration noise
                a = np.clip(a, -MAX_ACTION.cpu().numpy(), MAX_ACTION.cpu().numpy())
                # Reward shaping: exp(-pose_error)
                r = np.exp(-pose_error(a, flat_targets[i].cpu().numpy()))
                #r = -pose_error(a, flat_targets[i].cpu().numpy())
                s2 = flat_states[i+1].cpu().numpy()
                d = 0.0 if i < flat_states.size(0)-2 else 1.0
                replay_buffer.add(s, a, r, s2, d)

            # === DDPG update ===
            if replay_buffer.size < BATCH_REPLAY:
                continue
            s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(BATCH_REPLAY)

            # Critic update
            with torch.no_grad():
                a2 = actor_target(s2_batch)
                q2 = critic_target(s2_batch, a2)
                q_target = r_batch + (1 - d_batch) * gamma * q2
            q_pred = critic(s_batch, a_batch)
            critic_loss = nn.MSELoss()(q_pred, q_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), grad_clip)
            critic_optimizer.step()

            # Actor update
            actor_loss = -critic(s_batch, actor(s_batch)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), grad_clip)
            actor_optimizer.step()

            # Target network update
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Logging
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_reward += r_batch.mean().item()
            wandb.log({
                "batch/actor_loss": actor_loss.item(),
                "batch/critic_loss": critic_loss.item(),
                "batch/reward": r_batch.mean().item(),
                "epoch": epoch
            })
            pbar.set_postfix(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(), reward=r_batch.mean().item())

        # Epoch logging
        wandb.log({
            "epoch/actor_loss": total_actor_loss/len(train_loader),
            "epoch/critic_loss": total_critic_loss/len(train_loader),
            "epoch/reward": total_reward/len(train_loader),
            "epoch": epoch
        })

        # Validation (supervised, just for monitoring)
        if val_loader is not None:
            actor.eval()
            val_errors = []
            val_shaped_rewards = []
            with torch.no_grad():
                for batch in val_loader:
                    frames = batch["frames"].to(device)
                    poses  = batch["poses"].to(device)
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
                    flat_states = states.view(B*T, -1)
                    flat_targets = poses.view(B*T, 3)
                    pred_actions = actor(flat_states)
                    for i in range(flat_states.size(0)):
                        err = pose_error(pred_actions[i].cpu().numpy(), flat_targets[i].cpu().numpy())
                        val_errors.append(err)
                        val_shaped_rewards.append(np.exp(-err))

            avg_val_error = float(np.mean(val_errors))
            avg_val_shaped_reward = float(np.mean(val_shaped_rewards))
            print(f"[Validation] Avg Pose Error: {avg_val_error:.4f} | Avg Shaped Reward: {avg_val_shaped_reward:.4f}")
            wandb.log({
                "epoch/val_pose_error": avg_val_error,
                "epoch/val_shaped_reward": avg_val_shaped_reward,
                "epoch": epoch
            })
            if avg_val_error < best_val:
                best_val = avg_val_error
                torch.save(actor.state_dict(),
                           os.path.join(save_dir, f"best_actor_{epoch}.pt"))
                torch.save(critic.state_dict(),
                           os.path.join(save_dir, f"best_critic_{epoch}.pt"))

def main():
    CHECKPOINT_PATH = None  # or "/path/to/your/checkpoint.pt"
    DATASET_PATH    = "/data/students/earl/ai322/rldr/trajectories"
    SAVE_DIR        = "/data/students/earl/ai322/rldr/checkpoints"
    RUN_DIR         = os.path.join(SAVE_DIR, f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(RUN_DIR, exist_ok=True)

    wandb.init(
        project="ddpg_train_normal",
        dir=RUN_DIR,
        config={
            "history_length": HISTORY_LENGTH,
            "epochs":       NUM_EPOCHS,
            "batch_size":   BATCH_SIZE,
            "lr_actor":     LR_ACTOR,
            "lr_critic":    LR_CRITIC,
            "gamma":        GAMMA,
            "tau":          TAU,
        }
    )

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

    input_dim = 512 * (HISTORY_LENGTH + 1)
    actor = Actor(input_dim=input_dim, max_action=MAX_ACTION).to(device)
    actor_target = copy.deepcopy(actor).to(device)
    critic = Critic(input_dim=input_dim, action_dim=3).to(device)
    critic_target = copy.deepcopy(critic).to(device)

    if CHECKPOINT_PATH:
        actor.load_state_dict(torch.load(CHECKPOINT_PATH + "_actor.pt", map_location=device))
        actor_target.load_state_dict(torch.load(CHECKPOINT_PATH + "_actor.pt", map_location=device))
        critic.load_state_dict(torch.load(CHECKPOINT_PATH + "_critic.pt", map_location=device))
        critic_target.load_state_dict(torch.load(CHECKPOINT_PATH + "_critic.pt", map_location=device))

    train_ddpg(
        actor, actor_target, critic, critic_target, feature_extractor,
        train_loader, val_loader,
        NUM_EPOCHS, LR_ACTOR, LR_CRITIC,
        GAMMA, TAU, GRAD_CLIP, device, RUN_DIR
    )

if __name__ == "__main__":
    main()