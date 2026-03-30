import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .datasets.classification_dataset import load_data as load_classification_dataset
from .datasets.road_dataset import load_data as load_road_dataset
from .metrics import ConfusionMatrix


def train(
    weight_depth=10.0,
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("A")
    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_road_dataset("drive_data/train", transform_pipeline="aug", shuffle = True, batch_size = batch_size, num_workers = 2)
    val_data = load_road_dataset("drive_data/val", shuffle = False)

    # create loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}


    # Loss functions
    loss_seg = ClassificationLoss()
    loss_depth = torch.nn.L1Loss()

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()
        correct_train = 0
        total_train = 0

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logits, depth_pred = model(img)

            # Compute losses
            seg_loss = loss_seg(logits, logits)
            depth_loss = loss_depth(depth_pred, depth_pred)
            loss = seg_loss + weight_depth * depth_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
            logger.add_scalar("train/seg_loss", seg_loss.item(), global_step)
            logger.add_scalar("train/depth_loss", depth_loss.item(), global_step)
            logger.add_scalar("train/total_loss", loss.item(), global_step)
            
            total_seg_loss += seg_loss.item()
            total_depth_loss += depth_loss.item()
            total_loss += loss.item()
            global_step += 1

            predicted_labels = torch.argmax(out, dim=1)
            correct_train += (predicted_labels == label).sum().item()
            total_train += label.size(0)

            metrics["train_acc"].append(correct_train/total_train * 100)

            global_step += 1

         # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
    
            confusion = ConfusionMatrix(num_classes=3)  # For mIoU
            total_mae = 0.0
            total_mae_lane = 0.0
            n_pixels = 0
            n_lane_pixels = 0

            for batch in val_data:
                images = batch["image"].to(device)        # (B, 3, H, W)
                true_seg = batch["track"].to(device)      # (B, H, W)
                true_depth = batch["depth"].to(device)    # (B, H, W)

                # Get predictions
                pred_seg, pred_depth = model.predict(images)  # (B, H, W), (B, H, W)

                # --- mIoU ---
                confusion.update(pred_seg, true_seg)

                # --- Depth MAE (all pixels) ---
                abs_error = (pred_depth - true_depth).abs()
                total_mae += abs_error.sum().item()
                n_pixels += abs_error.numel()

                # --- Depth MAE (lane boundary pixels only) ---
                lane_mask = (true_seg != 0)  # class 1 and 2 are left/right boundary
                total_mae_lane += abs_error[lane_mask].sum().item()
                n_lane_pixels += lane_mask.sum().item()

            # Final metrics
            miou = confusion.get_mean_iou()
            depth_mae = total_mae / n_pixels
            depth_mae_lane = total_mae_lane / n_lane_pixels

            # Log metrics
            logger.add_scalar("val_mIoU", miou, global_step - 1)
            logger.add_scalar("val_depth_mae", depth_mae, global_step - 1)
            logger.add_scalar("val_depth_mae_lane", depth_mae_lane, global_step - 1)

            # Print progress
            if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:2d} / {num_epoch:2d}:\n"
                    f"  mIoU           = {miou:.4f}\n"
                    f"  Depth MAE      = {depth_mae:.4f}\n"
                    f"  Depth MAE (lane only) = {depth_mae_lane:.4f}"
                )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))