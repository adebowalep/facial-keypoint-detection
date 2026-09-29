"""Training entry point.

Note: the checkpoint shipped in ``models/`` was trained on a GPU workspace
and is committed to the repo specifically so this script does not need to be
re-run to use the model. Re-run it only if you intend to retrain (e.g. with
more data or a different architecture) and have GPU time available.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import REPO_ROOT
from .dataset import FacialKeypointsDataset
from .model import Net
from .transforms import Normalize, RandomCrop, Rescale, ToTensor


def build_data_transform() -> transforms.Compose:
    return transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])


def train(
    data_dir: Path,
    output_path: Path,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    data_transform = build_data_transform()

    train_dataset = FacialKeypointsDataset(
        csv_file=str(data_dir / "training_frames_keypoints.csv"),
        root_dir=str(data_dir / "training"),
        transform=data_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = Net().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    net.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            images = data["image"].type(torch.FloatTensor).to(device)
            key_pts = data["keypoints"].view(data["keypoints"].size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor).to(device)

            output_pts = net(images)
            loss = criterion(output_pts, key_pts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_i % 10 == 9:
                print(f"Epoch {epoch + 1}, Batch {batch_i + 1}, Avg. Loss: {running_loss / 1000:.6f}")
                running_loss = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), output_path)
    print(f"Saved checkpoint to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "keypoints_model_new.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
