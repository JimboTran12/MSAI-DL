from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):

    class ClassifierBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride, dropout_p: float):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.n1 = nn.GroupNorm(1, out_channels)
            self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.n2 = nn.GroupNorm(1, out_channels)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.drop = nn.Dropout2d(dropout_p)

            self.skip = (
                nn.Conv2d(in_channels, out_channels, 1, stride, 0)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x0):
            x = self.relu1(self.n1(self.c1(x0)))
            x = self.relu2(self.n2(self.c2(x)))
            x = self.skip(x0) + x
            return self.drop(x)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        num_blocks: int = 4,
        channels_l0: int = 32,
        dropout: float = 0.2,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))


        cnn_layers = [
            torch.nn.Conv2d(in_channels, channels_l0, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(dropout),
        ]
        c1 = channels_l0
        for _ in range(num_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.ClassifierBlock(c1, c2, stride=2, dropout_p=dropout))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        # Pool to one logit vector per image so output is (b, num_classes), not (b, c, h, w).
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
        cnn_layers.append(torch.nn.Flatten())

        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]


        return self.network(z)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        C = base_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(in_channels, C)          # (B, C, H, W)
        self.pool1 = nn.Conv2d(C, C, kernel_size=2, stride=2)   # H/2

        self.enc2 = conv_block(C, C * 2)                # (B, 2C, H/2, W/2)
        self.pool2 = nn.Conv2d(C * 2, C * 2, kernel_size=2, stride=2)  # H/4

        self.enc3 = conv_block(C * 2, C * 4)            # (B, 4C, H/4, W/4)

        # Bottleneck with dilation to increase receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(1, C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * 4, C * 4, kernel_size=3, padding=1),
            nn.GroupNorm(1, C * 4),
            nn.ReLU(inplace=True),
        )

        # Decoder: bilinear upsample + conv after concat with skip
        self.dec1 = conv_block(C * 4 + C * 2, C * 2)  # after concat with enc2 (H/4 -> H/2)
        self.dec2 = conv_block(C * 2 + C, C)          # after concat with enc1 (H/2 -> H)

        # Heads
        self.seg_head = nn.Conv2d(C, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # encoding
        e1 = self.enc1(z)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)

        # bottleneck
        b = self.bottleneck(e3)

        # decoding with skip connections
        u1 = nn.functional.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        
        logits = self.seg_head(d2)
        depth = self.depth_head(d2).squeeze(1)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
