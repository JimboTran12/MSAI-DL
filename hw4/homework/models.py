from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.linear1 = torch.nn.Linear(4 * n_track, 128)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 64)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(64, 2 * n_waypoints)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        track_full = torch.cat([track_left, track_right], dim=-1).view(track_left.shape[0], 4 * self.n_track)

        return self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(track_full))))).view(track_right.shape[0], self.n_waypoints, 2)




class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(2 * n_waypoints, d_model)
        self.key_embed = nn.Linear(4, d_model)
        self.value_embed = nn.Linear(4, d_model)

        self.attention = torch.nn.MultiheadAttention(d_model, 8, batch_first = True)
        self.output = nn.Linear(d_model, 2 * n_waypoints)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        track_full = torch.cat([track_left, track_right], dim=-1)
        k = self.key_embed(track_full)
        v = self.value_embed(track_full)

        q = self.query_embed.weight.unsqueeze(0).repeat(track_full.size(0), 1, 1)
        attend, _ = self.attention(q, k, v)

        return self.output(attend.mean(dim = 1)).view(attend.shape[0], self.n_waypoints, 2)


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        cnn_layers = [
          torch.nn.Conv2d(3, 16, kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(16),
          torch.nn.ReLU(),

          torch.nn.Conv2d(16, 32, kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),

          torch.nn.Conv2d(32, 64, kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),

          torch.nn.Conv2d(64, 128, kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),

          torch.nn.Conv2d(128, 256, kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(256),
          torch.nn.ReLU(),

          torch.nn.Conv2d(256, 512, kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(512),
          torch.nn.ReLU(),

          torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),

          torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),

          torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),

          torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),

          torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),

          torch.nn.ConvTranspose2d(16, self.n_waypoints * 2, kernel_size=2, stride=2),
          torch.nn.ReLU()
        ]
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        return self.network(x).view(image.shape[0], -1, self.n_waypoints, 2).mean(dim=1)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
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


def save_model(model: torch.nn.Module) -> str:
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


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024