import argparse
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
from .models import load_model, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    num_workers: int = 4,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    pin_memory = use_cuda

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name, **kwargs)
    model = model.to(device)

    train_data = load_data("drive_data/train", transform_pipeline="aug", return_dataloader=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = load_data(
        "drive_data/val",
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    print("Computing class weights for segmentation loss...")
    num_classes = 3
    class_counts = torch.zeros(num_classes, dtype=torch.float)
    for i in range(len(train_data)):
        track = train_data[i]["track"]
        if not isinstance(track, torch.Tensor):
            track = torch.from_numpy(np.array(track))
        class_counts += torch.bincount(track.flatten(), minlength=num_classes).float()

    eps = 1e-6
    seg_weights = class_counts.sum() / (class_counts + eps)
    seg_weights = seg_weights / seg_weights.mean()
    print(f"Segmentation loss class weights: {seg_weights.tolist()}")

    seg_loss_func = torch.nn.CrossEntropyLoss(weight=seg_weights.to(device))
    depth_loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=use_cuda)

    train_metric = DetectionMetric()
    val_metric = DetectionMetric()

    def to_device(batch):
        img = batch["image"].to(device, non_blocking=pin_memory)
        depth = batch["depth"].to(device, non_blocking=pin_memory)
        track = batch["track"].to(device, non_blocking=pin_memory).long()
        return img, depth, track

    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()

        epoch_bg_count = 0
        epoch_total_count = 0

        for data in train_dataloader:
            img, depth, track = to_device(data)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_cuda):
                logits, depth_pred = model(img)
                seg_loss = seg_loss_func(logits, track)
                depth_loss = depth_loss_func(depth_pred, depth)
                loss = seg_loss + depth_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                batch_bg = int((pred == 0).sum().item())
                batch_total = int(pred.numel())
                epoch_bg_count += batch_bg
                epoch_total_count += batch_total
                train_metric.add(pred, track, depth_pred, depth)

        epoch_stats = train_metric.compute()
        epoch_bg_pct = epoch_bg_count / epoch_total_count * 100.0 if epoch_total_count > 0 else 0.0

        val_metric.reset()
        val_bg_count = 0
        val_total_count = 0
        model.eval()

        with torch.inference_mode():
            for data in val_dataloader:
                img, depth, track = to_device(data)

                with autocast(enabled=use_cuda):
                    logits, depth_pred = model(img)

                pred = logits.argmax(dim=1)
                batch_bg = int((pred == 0).sum().item())
                batch_total = int(pred.numel())
                val_bg_count += batch_bg
                val_total_count += batch_total
                val_metric.add(pred, track, depth_pred, depth)

            epoch_val = val_metric.compute()
            epoch_val_bg_pct = val_bg_count / val_total_count * 100.0 if val_total_count > 0 else 0.0

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"iou={epoch_stats['iou']:.4f} "
            f"accuracy={epoch_stats['accuracy']:.4f} "
            f"bg%={epoch_bg_pct:.2f} "
            f"val_iou={epoch_val['iou']:.4f} "
            f"val_accuracy={epoch_val['accuracy']:.4f} "
            f"val_bg%={epoch_val_bg_pct:.2f}"
        )

    save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=4)

    train(**vars(parser.parse_args()))
