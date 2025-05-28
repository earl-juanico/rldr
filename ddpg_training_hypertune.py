import os
import datetime
import itertools
import time
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import wandb
from tqdm import tqdm
import numpy as np
import copy

# === Config ===
NUM_FRAMES     = 32
NUM_EPOCHS     = 150
BATCH_SIZE     = 16
LR_ACTOR       = 1e-4
POS_WEIGHT     = 1.0
ORI_WEIGHT     = 2.0
REPLAY_SIZE    = 100_000
REPLAY_INIT    = 1000
BATCH_REPLAY   = 128
MAX_ACTION     = torch.tensor([0.5, 0.5, 1.0])  # x, y in meters, theta normalized to [-1, 1]
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# === Feature extractor ===
class FeatureExtractor(torch.nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
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
    return λ_pos * pos_err**2 + λ_ori * (ori_err)**2

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
            torch.FloatTensor(np.array([self.state[i] for i in idxs])),
            torch.FloatTensor(np.array([self.action[i] for i in idxs])),
            torch.FloatTensor(np.array([self.reward[i] for i in idxs])).unsqueeze(1),
            torch.FloatTensor(np.array([self.next_state[i] for i in idxs])),
            torch.FloatTensor(np.array([self.done[i] for i in idxs])).unsqueeze(1)
        )

class Actor(torch.nn.Module):
    def __init__(self, input_dim, action_dim=3, max_action=MAX_ACTION):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim)
        )
        self.max_action = max_action

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.max_action.to(x.device)

class Critic(torch.nn.Module):
    def __init__(self, input_dim, action_dim=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + action_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

def train_ddpg_hparams(hparams, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Unpack hyperparameters
    LR_CRITIC     = hparams["lr_critic"]
    GAMMA         = hparams["gamma"]
    GRAD_CLIP     = hparams["grad_clip"]
    HISTORY_LENGTH= hparams["history_length"]
    run_name      = f"lrC{LR_CRITIC}-g{GAMMA}-c{GRAD_CLIP}-h{HISTORY_LENGTH}"

    # Logging
    wandb.init(
        project="ddpg_hypertune_2",
        name=run_name,
        config=hparams
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

    ds = PoseTrajectoryDataset(hparams["data_path"], transform=val_t, history=HISTORY_LENGTH)
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
    actor     = Actor(input_dim=input_dim).to(device)
    actor_target = copy.deepcopy(actor).to(device)
    critic    = Critic(input_dim=input_dim).to(device)
    critic_target = copy.deepcopy(critic).to(device)

    # Add these lines here:
    wandb.watch(actor, log="all", log_freq=100)
    wandb.watch(critic, log="all", log_freq=100)
    
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
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
                #r = np.exp(-pose_error(a, flat_targets[i])) # exponential shaped reward
                r = -pose_error(a, flat_targets[i])
                s2 = flat_states[i+1]
                d = 0.0 if i < flat_states.shape[0]-2 else 1.0
                replay_buffer.add(s, a, r, s2, d)
                if replay_buffer.size >= REPLAY_INIT:
                    break
            if replay_buffer.size >= REPLAY_INIT:
                break

    for epoch in range(1, NUM_EPOCHS+1):
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
                #r = np.exp(-pose_error(a, flat_targets[i].cpu().numpy()))  # exponential shaped reward
                r = -pose_error(a, flat_targets[i].cpu().numpy())
                s2 = flat_states[i+1].cpu().numpy()
                d = 0.0 if i < flat_states.size(0)-2 else 1.0
                replay_buffer.add(s, a, r, s2, d)

            # === DDPG update ===
            if replay_buffer.size < BATCH_REPLAY:
                continue
            s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(BATCH_REPLAY)
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            r_batch = r_batch.to(device)
            s2_batch = s2_batch.to(device)
            d_batch = d_batch.to(device)

            # Critic update
            with torch.no_grad():
                a2 = actor_target(s2_batch)
                q2 = critic_target(s2_batch, a2)
                q_target = r_batch + (1 - d_batch) * GAMMA * q2
            q_pred = critic(s_batch, a_batch)
            critic_loss = torch.nn.MSELoss()(q_pred, q_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            critic_optimizer.step()

            # Actor update
            actor_loss = -critic(s_batch, actor(s_batch)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), GRAD_CLIP)
            actor_optimizer.step()

            # Target network update
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

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
                           os.path.join(hparams["save_dir"], f"best_actor_{epoch}.pt"))
                torch.save(critic.state_dict(),
                           os.path.join(hparams["save_dir"], f"best_critic_{epoch}.pt"))

    wandb.finish()

def get_visible_gpus():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return [int(x) for x in cuda_visible.split(",") if x.strip().isdigit()]
    else:
        # If not set, use all GPUs
        return list(range(torch.cuda.device_count()))

def main():
    DATASET_PATH = "/data/students/earl/ai322/rldr/trajectories"
    SAVE_DIR     = "/data/students/earl/ai322/rldr/checkpoints"
    root_save_dir = os.path.join(
        SAVE_DIR, f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )
    os.makedirs(root_save_dir, exist_ok=True)

    # Hyperparameter grid
    lr_critic_list    = [1e-4, 5e-4, 1e-3]
    gamma_list        = [0.95, 0.99]
    grad_clip_list    = [0.5, 1.0, 2.0]
    history_length_list = [2, 3, 4]

    # Fixed settings
    common = {
        "data_path":  DATASET_PATH,
        "save_dir":   root_save_dir,
    }

    grid = list(itertools.product(
        lr_critic_list, gamma_list, grad_clip_list, history_length_list
    ))

    jobs = []
    for lr_c, gamma, grad_clip, hist in grid:
        run_name = f"lrC{lr_c}-g{gamma}-c{grad_clip}-h{hist}"
        run_dir = os.path.join(root_save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        hparams = dict(common,
            lr_critic=lr_c,
            gamma=gamma,
            grad_clip=grad_clip,
            history_length=hist,
            save_dir=run_dir
        )
        jobs.append(hparams)

    visible_gpus = get_visible_gpus()
    max_concurrent = len(visible_gpus)
    active = []

    while jobs or active:
        # Find available GPUs
        busy = {gpu for _, gpu in active}
        candidates = [gpu for gpu in visible_gpus if gpu not in busy]

        while candidates and jobs and len(active) < max_concurrent:
            hparams = jobs.pop(0)
            gpu = candidates.pop(0)
            p = mp.Process(target=train_ddpg_hparams, args=(hparams, gpu))
            p.start()
            active.append((p, gpu))

        time.sleep(5)
        for p, gpu in active.copy():
            if not p.is_alive():
                p.join()
                active.remove((p, gpu))

    print("All hyperparameter runs completed.")

if __name__ == "__main__":
    #mp.set_start_method('spawn', force=True)
    mp.set_start_method('forkserver', force=True)
    main()