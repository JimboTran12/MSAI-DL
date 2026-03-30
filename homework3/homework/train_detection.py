import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 10,
    lr: float = 5e-2,
    batch_size: int = 64,
    seed: int = 2024,
    depth_loss_weight: float = 1.0,
    train_path: str = "drive_data/train",
    val_path: str = "drive_data/val",
    **kwargs,
):
    """
    Training loop for the Detector (segmentation + depth), following the homework 2 pattern:
    model, losses, optimizer, train/val data, epochs, logging, save_model.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)

    train_data = load_data(
        train_path,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline="default",
    )
    val_data = load_data(
        val_path,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline="default",
    )

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0

    def seg_accuracy(logits: torch.Tensor, track: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return (pred == track).float().mean().item()

    def depth_mae(pred: torch.Tensor, depth: torch.Tensor) -> float:
        return (pred - depth).abs().mean().item()

    for epoch in range(num_epoch):
        model.train()
        train_seg_acc = []
        train_depth_mae = []
        train_losses = []

        for batch in train_data:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            track = batch["track"].to(device).long()

            optim.zero_grad()
            logits, raw_depth = model(img)

            loss_seg = F.cross_entropy(logits, track)
            loss_depth = F.l1_loss(raw_depth, depth)
            loss = loss_seg + depth_loss_weight * loss_depth

            loss.backward()
            optim.step()

            global_step += 1
            train_losses.append(loss.item())
            train_seg_acc.append(seg_accuracy(logits.detach(), track))
            train_depth_mae.append(depth_mae(raw_depth.detach(), depth))

            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/loss_seg", loss_seg.item(), global_step)
            logger.add_scalar("train/loss_depth", loss_depth.item(), global_step)

        with torch.inference_mode():
            model.eval()
            val_seg_acc = []
            val_depth_mae = []

            for batch in val_data:
                img = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device).long()

                logits, raw_depth = model(img)
                val_seg_acc.append(seg_accuracy(logits, track))
                val_depth_mae.append(depth_mae(raw_depth, depth))

        epoch_train_loss = float(np.mean(train_losses))
        epoch_train_seg = float(np.mean(train_seg_acc))
        epoch_train_depth = float(np.mean(train_depth_mae))
        epoch_val_seg = float(np.mean(val_seg_acc))
        epoch_val_depth = float(np.mean(val_depth_mae))

        logger.add_scalar("epoch/train_loss", epoch_train_loss, epoch)
        logger.add_scalar("epoch/train_seg_acc", epoch_train_seg, epoch)
        logger.add_scalar("epoch/train_depth_mae", epoch_train_depth, epoch)
        logger.add_scalar("epoch/val_seg_acc", epoch_val_seg, epoch)
        logger.add_scalar("epoch/val_depth_mae", epoch_val_depth, epoch)

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={epoch_train_loss:.4f} "
            f"train_seg_acc={epoch_train_seg:.4f} "
            f"train_depth_mae={epoch_train_depth:.4f} | "
            f"val_seg_acc={epoch_val_seg:.4f} "
            f"val_depth_mae={epoch_val_depth:.4f}"
        )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--depth_loss_weight", type=float, default=1.0)
    parser.add_argument("--train_path", type=str, default="drive_data/train")
    parser.add_argument("--val_path", type=str, default="drive_data/val")

    train(**vars(parser.parse_args()))
