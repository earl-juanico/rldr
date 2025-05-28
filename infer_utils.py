"""
-----------------
infer_utils.py
-----------------

Based on ddpg_training.py (with normalized angles)

Utility functions for inference in a machine learning model.
This module contains functions to load a model, preprocess input data,
perform inference, and postprocess the output.

The functions are copied from the training code.

"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import os


GRID_SIZE      = 5
CELL_SIZE      = 0.3
NUM_ORIENT     = 72
NUM_FRAMES     = 32
NUM_ACTIONS    = GRID_SIZE * GRID_SIZE * NUM_ORIENT

# === Config (copy from ddpg_training.py) ===
NUM_FRAMES     = 32
HISTORY_LENGTH = 2
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
MAX_ACTION     = torch.tensor([0.75, 0.75, 1.0])  # x, y in meters, theta normalized to [-1, 1]

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

def action_to_pose(action_id):
    ang = action_id % NUM_ORIENT
    idx = action_id // NUM_ORIENT
    row, col = divmod(idx, GRID_SIZE)
    x = (col - GRID_SIZE//2)*CELL_SIZE
    y = (row - GRID_SIZE//2)*CELL_SIZE
    θ = ang*(360.0/NUM_ORIENT)
    return [x, y, θ]


def infer_actions(image_paths, model, feature_extractor, device):
    assert len(image_paths) == NUM_FRAMES, f"Expected {NUM_FRAMES} images"
    frames = preprocess_images(image_paths).unsqueeze(0).to(device)  # (1, T, C, H, W)
    B, T, C, H, W = frames.shape
    imgs = frames.view(B*T, C, H, W)
    feats = feature_extractor(imgs).view(B, T, -1)  # (1, T, 512)

    # Build state vectors as in training
    if HISTORY_LENGTH == 1:
        f0 = feats[:, :1, :].expand(-1, T, -1)
        states = torch.cat([f0, feats], dim=-1)
    else:
        f0 = feats[:, :1, :].expand(-1, T, -1)
        all_s = []
        for t in range(T):
            parts = [f0[:, t]]
            for k in range(HISTORY_LENGTH-1, 0, -1):
                parts.append(feats[:, t-k] if t-k>=0 else feats[:, 0])
            parts.append(feats[:, t])
            all_s.append(torch.cat(parts, dim=-1))
        states = torch.stack(all_s, dim=1)  # (1, T, (H+1)*512)

    flat = states.view(B*T, -1)
    logits, _ = model(flat)
    action_ids = torch.argmax(logits, dim=1).cpu().numpy()  # (T,)

    poses = [action_to_pose(int(a)) for a in action_ids]
    return action_ids, poses

def load_model(checkpoint_path, device, history_length=HISTORY_LENGTH):
    input_dim = 512 * (HISTORY_LENGTH + 1)
    model = FeatureActorCriticNet(input_dim=input_dim, num_actions=NUM_ACTIONS).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


# === Model definitions (copy from ddpg_training.py) ===
'''
class FeatureActorCriticNet(nn.Module):
    def __init__(self, input_dim=512, num_actions=NUM_ACTIONS, use_dropout=True, dropout_p=0.2):
        super().__init__()

        def make_head(output_dim, final_layer_size):
            layers = [
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(p=dropout_p) if use_dropout else nn.Identity(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(p=dropout_p) if use_dropout else nn.Identity(),
                nn.Linear(256, final_layer_size)
            ]
            return nn.Sequential(*layers)

        self.actor = make_head(output_dim=num_actions, final_layer_size=num_actions)
        self.critic = make_head(output_dim=1, final_layer_size=1)

    def forward(self, x):
        return self.actor(x), self.critic(x)
'''

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

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim=3, max_action=MAX_ACTION):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.max_action = max_action

    def forward(self, x):
        # output x,y in meters, theta in [-1,1]
        return torch.tanh(self.net(x)) * self.max_action.to(x.device)

# === Utilities ===
def load_actor(checkpoint_path, device, history_length=HISTORY_LENGTH):
    input_dim = 512 * (history_length + 1)
    actor = Actor(input_dim=input_dim).to(device)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    actor.eval()
    return actor

def load_feature_extractor(device):
    extractor = FeatureExtractor(fine_tune=False).to(device)
    extractor.eval()
    return extractor

def preprocess_images(image_paths):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    imgs = [t(Image.open(p).convert("RGB")) for p in image_paths]
    return torch.stack(imgs)  # (T, C, H, W)

def infer_poses(image_paths, actor, feature_extractor, device, history_length=HISTORY_LENGTH):
    """
    Returns a list of [x, y, theta_deg] for each frame.
    Theta is recovered from the model’s normalized [-1,1] output.
    """
    assert len(image_paths) == NUM_FRAMES, f"Expected {NUM_FRAMES} images"
    frames = preprocess_images(image_paths).unsqueeze(0).to(device)  # (1, T, C, H, W)
    B, T, C, H, W = frames.shape
    imgs = frames.view(B*T, C, H, W)
    feats = feature_extractor(imgs).view(B, T, -1)  # (1, T, 512)

    # build history‐stacked state as in training
    if history_length == 1:
        f0 = feats[:, :1, :].expand(-1, T, -1)
        states = torch.cat([f0, feats], dim=-1)
    else:
        f0 = feats[:, :1, :].expand(-1, T, -1)
        all_s = []
        for t in range(T):
            parts = [f0[:, t]]
            for k in range(history_length-1, 0, -1):
                parts.append(feats[:, t-k] if t-k>=0 else feats[:, 0])
            parts.append(feats[:, t])
            all_s.append(torch.cat(parts, dim=-1))
        states = torch.stack(all_s, dim=1)  # (1, T, (H+1)*512)

    flat = states.view(B*T, -1)
    with torch.no_grad():
        out = actor(flat).cpu().numpy()  # (T,3): [x, y, theta_norm]
    poses_deg = []
    for x, y, theta_norm in out:
        theta_deg = float(theta_norm) * 180.0   # map [-1,1] → [-180,180]
        poses_deg.append([float(x), float(y), theta_deg])
    return poses_deg