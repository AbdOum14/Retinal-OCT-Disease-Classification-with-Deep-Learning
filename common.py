"""Shared utilities for train.py, evaluate.py and the notebooks."""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use CUDA GPU if available, then Apple Silicon (MPS), otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_transforms(img_size: int = 224):
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfm, eval_tfm


class OCTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, tfm):
        self.paths = df["path"].tolist()
        self.labels = [class_to_idx[x] for x in df["label"]]
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # grayscale OCT -> 3 channels
        return self.tfm(img), self.labels[i]


def build_model(num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    """ResNet18. If freeze=True, only layer4 and the new final layer are trainable."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new layer: trainable
    if freeze:
        for p in model.layer4.parameters():
            p.requires_grad = True
    return model


def grad_cam(model, x, target_layer, class_idx=None):
    """Minimal Grad-CAM on the last convolutional block (layer4 of ResNet18).

    Returns (heatmap normalised to 0-1, class index used).
    The model must be in eval() mode and target_layer must have gradients enabled.
    """
    acts, grads = [], []

    def fwd_hook(_m, _i, out):
        acts.append(out)
        out.register_hook(lambda g: grads.append(g))

    handle = target_layer.register_forward_hook(fwd_hook)
    model.zero_grad()
    logits = model(x)
    if class_idx is None:
        class_idx = int(logits.argmax(1).item())
    logits[0, class_idx].backward()
    handle.remove()

    a, g = acts[0], grads[0]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * a).sum(1)).squeeze(0)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), class_idx
