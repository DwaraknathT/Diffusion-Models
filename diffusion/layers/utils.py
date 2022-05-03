import torch


def _grot90(
    x: torch.tensor,
    k: int,
) -> torch.tensor:
    # rotate the channels of the filter and cyclically
    # permute them
    return torch.rot90(x.roll(k, 2), k, (3, 4))
