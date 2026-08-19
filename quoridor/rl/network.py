import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class QuoridorNet(nn.Module):
    """CNN trunk with two heads, matching the AlphaZero-style design: a
    policy over the fixed action space (moves + wall placements) to guide
    search, and a value over win probability per player so leaf positions
    can be scored without rolling out to a terminal state."""

    def __init__(
        self,
        size: int,
        player_count: int,
        in_channels: int,
        action_size: int,
        trunk_channels: int = 64,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(trunk_channels) for _ in range(num_res_blocks)))

        policy_channels = 2
        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_channels * size * size, action_size),
        )

        value_channels = 1
        value_hidden = 64
        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_channels * size * size, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, player_count),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(self.stem(x))
        return self.policy_head(features), self.value_head(features)

    def predict(
        self, x: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched inference: legal_mask is (batch, action_size) bool,
        True where the action is legal in that sample's position."""
        policy_logits, value_logits = self.forward(x)
        policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
        policy = F.softmax(policy_logits, dim=-1)
        win_probs = F.softmax(value_logits, dim=-1)
        return policy, win_probs
