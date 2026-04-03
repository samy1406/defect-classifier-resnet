import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir="data", batch_size=32):
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(data_dir + "/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(data_dir + "/val",   transform=val_transform)
    test_dataset  = datasets.ImageFolder(data_dir + "/test",  transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Inspect one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")