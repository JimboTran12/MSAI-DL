import torch
import argparse
import numpy as np
from .models import Detector, load_model, save_model
from .metrics import DetectionMetric
from .datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # load and augment data
    train_data = load_data("drive_data/train", return_dataloader=False)
    train_data_aug = load_data("drive_data/train", transform_pipeline="aug", return_dataloader=False)
    train_data_full = torch.utils.data.ConcatDataset([train_data, train_data_aug])

    train_dataloader = torch.utils.data.DataLoader(train_data_full, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = load_data("drive_data/val", shuffle=False)

    # handle class imbalance
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

    # loss function and optimizer
    seg_loss_func = torch.nn.CrossEntropyLoss(weight=seg_weights.to(device))
    depth_loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_metric = DetectionMetric()
    val_metric = DetectionMetric()

    # training loop 
    for epoch in range(num_epoch):
        train_metric.reset()


        epoch_bg_count = 0
        epoch_total_count = 0

        for data in train_dataloader:
            img, depth, track = data["image"].to(device), data["depth"].to(device), data["track"].to(device)

            optimizer.zero_grad()
            logits, depth_pred = model(img)

            seg_loss = seg_loss_func(logits, track)
            depth_loss = depth_loss_func(depth_pred, depth)

            # simple combination of losses 
            loss = seg_loss + depth_loss
            loss.backward()
            optimizer.step()

            pred, raw_depth = model.predict(img)
            batch_bg = int((pred == 0).sum().item())
            batch_total = int(pred.numel())

            epoch_bg_count += batch_bg
            epoch_total_count += batch_total

            # # detection metrics
            train_metric.add(pred, track, depth_pred, depth)

            # batch_train_acc = (pred == label).float().mean().item()
            # metrics["train_acc"].append(batch_train_acc)
        
        epoch_stats = train_metric.compute()
        epoch_bg_pct = epoch_bg_count / epoch_total_count * 100.0 if epoch_total_count > 0 else 0.0
        

        # validation
        val_metric.reset()
        val_bg_count = 0
        val_total_count = 0
        model.eval()

        with torch.inference_mode():
            for data in val_dataloader:
                img, depth, track = data["image"].to(device), data["depth"].to(device), data["track"].to(device)

                logits, depth_pred = model(img)

                pred, raw_depth = model.predict(img)

                batch_bg = int((pred == 0).sum().item())
                batch_total = int(pred.numel())

                val_bg_count += batch_bg
                val_total_count += batch_total

                val_metric.add(pred, track, depth_pred, depth)
            
            epoch_val = val_metric.compute()
            epoch_val_bg_pct = val_bg_count / val_total_count * 100.0 if val_total_count > 0 else 0.0
        
        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"iou={epoch_stats['iou']:.4f} "
                f"accuracy={epoch_stats['accuracy']:.4f} "
                # f"abs_depth_error={epoch_stats['abs_depth_error']:.4f} "
                # f"tp_depth_error={epoch_stats['tp_depth_error']:.4f} "
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
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))