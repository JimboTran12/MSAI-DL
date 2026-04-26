import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "linear_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    metrics = {"train_L1_err": [], "train_long_err": [], "train_lat_err": [], "val_L1_err": [], "val_long_err": [], "val_lat_err": []}
    global_step = 0
    train_planner = PlannerMetric()
    val_planner = PlannerMetric()
    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
          metrics[key].clear()
        
        train_planner.reset()
        val_planner.reset()

        model.train()

        for _, x in enumerate(train_data):
            image, track_left, track_right, waypoints, waypoints_mask = x['image'].to(device), x['track_left'].to(device), x['track_right'].to(device), x['waypoints'].to(device), x['waypoints_mask'].to(device)
            # TODO: implement training step
            pred_logits = None
            if model_name == "cnn_planner":
              pred_logits = model(image)
            else:
              pred_logits = model(track_left, track_right)
            loss_val = loss_func(pred_logits, waypoints)
            train_planner.add(pred_logits, waypoints, waypoints_mask)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1
        train_planner_computation = train_planner.compute()
        metrics["train_L1_err"].append(train_planner_computation["l1_error"])
        metrics["train_long_err"].append(train_planner_computation["longitudinal_error"])
        metrics["train_lat_err"].append(train_planner_computation["lateral_error"])

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            
            for _, x in enumerate(val_data):
                image, track_left, track_right, waypoints, waypoints_mask = x['image'].to(device), x['track_left'].to(device), x['track_right'].to(device), x['waypoints'].to(device), x['waypoints_mask'].to(device)

                # TODO: compute validation accuracy
                
                if model_name == "cnn_planner":
                  pred_logits = model(image)
                else:
                  pred_logits = model(track_left, track_right)
                val_planner.add(pred_logits, waypoints, waypoints_mask)
        val_planner_computation = val_planner.compute()
        metrics["val_L1_err"].append(val_planner_computation["l1_error"])
        metrics["val_long_err"].append(val_planner_computation["longitudinal_error"])
        metrics["val_lat_err"].append(val_planner_computation["lateral_error"])

        # log average train and val accuracy to tensorboard
        epoch_train_L1_err = torch.as_tensor(metrics["train_L1_err"]).mean()
        epoch_train_long_err = torch.as_tensor(metrics["train_long_err"]).mean()
        epoch_train_lat_err = torch.as_tensor(metrics["train_lat_err"]).mean()
        epoch_val_L1_err = torch.as_tensor(metrics["val_L1_err"]).mean()
        epoch_val_long_err = torch.as_tensor(metrics["val_long_err"]).mean()
        epoch_val_lat_err = torch.as_tensor(metrics["val_lat_err"]).mean()

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}:\n"
                f"train_L1_err={epoch_train_L1_err:.4f}\n"
                f"train_long_err={epoch_train_long_err:.4f}\n"
                f"train_lat_err={epoch_train_lat_err:.4f}\n"
                f"val_L1_err={epoch_val_L1_err:.4f}\n"
                f"val_long_err={epoch_val_long_err:.4f}\n"
                f"val_lat_err={epoch_val_lat_err:.4f}\n"
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