"""Train a ResNet-18 transfer-learning classifier (stop / not_stop)."""
import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(n_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # freeze the feature extractor
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs, device):
    loss_list, accuracy_list = [], []
    n_val = len(val_loader.dataset)
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(n_epochs)):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        loss_list.append(float(np.mean(losses)))

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
        acc = correct / n_val
        accuracy_list.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}: lr={optimizer.param_groups[0]['lr']:.5f} "
            f"train_loss={loss_list[-1]:.4f} val_acc={acc:.4f}"
        )

    model.load_state_dict(best_wts)
    return accuracy_list, loss_list, model


def plot_curves(loss, acc, path):
    fig, ax1 = plt.subplots()
    ax1.plot(loss, color="tab:red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(acc, color="tab:blue")
    ax2.set_ylabel("Validation accuracy", color="tab:blue")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--base-lr", type=float, default=0.001)
    parser.add_argument("--max-lr", type=float, default=0.01)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--output", default="model.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = ImageFolder(f"{args.data_dir}/train", transform=transform)
    val_ds = ImageFolder(f"{args.data_dir}/val", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = build_model(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = None
    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
            step_size_up=5, mode="triangular2",
        )

    acc, loss, model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device
    )
    torch.save(model.state_dict(), args.output)
    plot_curves(loss, acc, "training_curves.png")
    print(f"Saved {args.output} (best val acc {max(acc):.4f}) and training_curves.png")


if __name__ == "__main__":
    main()
