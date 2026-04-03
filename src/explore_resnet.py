import torchvision.models as models
import torch

# Load pretrained ResNet18
model = models.resnet18(weights="IMAGENET1K_V1")
print(model)

# How many total parameters?
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")

# What does the final fc layer look like?
print(f"\nOriginal FC layer: {model.fc}")
print(f"FC input features: {model.fc.in_features}")
print(f"FC output classes: {model.fc.out_features}")

import torch.nn as nn

NUM_CLASSES = 6  # PCB defect classes

# Step 1 — Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Step 2 — Replace the FC head (this is automatically trainable)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Step 3 — Check what's trainable now
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"\nNew FC layer: {model.fc}")