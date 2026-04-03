import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=6, mode="feature_extract"):
    # Load pretrained ResNet18
    model = models.resnet18(weights="IMAGENET1K_V1")

    if mode == "feature_extract":
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

    elif mode == "finetune":
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4 only
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace FC head for both modes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")


if __name__ == "__main__":
    print("--- Feature Extraction Mode ---")
    model = get_model(num_classes=6, mode="feature_extract")
    print_trainable_params(model)

    print("\n--- Fine-tune Mode ---")
    model = get_model(num_classes=6, mode="finetune")
    print_trainable_params(model)