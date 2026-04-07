import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from src.model import get_model
from src.dataset import get_dataloaders


def evaluate(model_path, mode="finetune"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader, classes = get_dataloaders(batch_size=32)
    num_classes = len(classes)

    # Load model
    model = get_model(num_classes=num_classes, mode=mode)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nTest Accuracy: {acc:.2f}%")

    # Per class accuracy
    print("\nPer-class Accuracy:")
    for i, cls in enumerate(classes):
        mask = all_labels == i
        cls_acc = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        print(f"  {cls:20s}: {cls_acc:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {mode} (Test Acc: {acc:.2f}%)")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{mode}.png", dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved as confusion_matrix_{mode}.png")


if __name__ == "__main__":
    evaluate("best_model_finetune.pth", mode="finetune")