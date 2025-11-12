import torch

# Indices for PushT state representation
# - In the raw (angle) representation, the angle is at index 4
# - In the sin/cos representation, sin is at 4 and cos is at 5
ANGLE_INDEX = 4
SINCOS_IDXS = (4, 5)


def angle_to_sincos(x: torch.Tensor, angle_idx: int = ANGLE_INDEX) -> torch.Tensor:
    """
    Replace the angle at `angle_idx` with its sin and cos components.
    Works with tensors of shape (..., D); operates on the last dimension.
    Returns a tensor with shape (..., D+1).
    """
    theta = x[..., angle_idx]
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    left = x[..., :angle_idx]
    right = x[..., angle_idx + 1 :]
    return torch.cat((left, sin_theta, cos_theta, right), dim=-1)


def sincos_to_angle(
    x: torch.Tensor,
    sin_idx: int = SINCOS_IDXS[0],
    cos_idx: int = SINCOS_IDXS[1],
) -> torch.Tensor:
    """
    Replace sin/cos at (sin_idx, cos_idx) with a single angle using atan2(sin, cos).
    Works with tensors of shape (..., D); operates on the last dimension.
    Returns a tensor with shape (..., D-1).
    """
    sin = x[..., sin_idx]
    cos = x[..., cos_idx]
    theta = torch.atan2(sin, cos).unsqueeze(-1)

    # Gather all columns except sin/cos
    D = x.size(-1)
    keep = [i for i in range(D) if i not in (sin_idx, cos_idx)]
    parts = [x[..., i : i + 1] for i in keep]
    insert_pos = min(sin_idx, cos_idx)
    parts.insert(insert_pos, theta)
    return torch.cat(parts, dim=-1)

