import torch


def gaussian_shuffle(x):
    """Pixel-level Gaussian shuffling along H and W dimensions.

    Args:
        x: tensor(N, C, H, W)
    Returns:
        shuffled: tensor(N, C, H, W) with rows and columns randomly permuted
        h_idx: permutation indices for H dimension
        w_idx: permutation indices for W dimension
    """
    _, _, H, W = x.shape
    h_idx = torch.randperm(H, device=x.device)
    w_idx = torch.randperm(W, device=x.device)
    shuffled = x[:, :, h_idx, :]
    shuffled = shuffled[:, :, :, w_idx]
    return shuffled, h_idx, w_idx


def gaussian_inverse_shuffle(x, h_idx, w_idx):
    """Inverse of gaussian_shuffle: restores original spatial order.

    Args:
        x: tensor(N, C, H, W) — shuffled image
        h_idx: permutation indices used for H dimension
        w_idx: permutation indices used for W dimension
    Returns:
        restored: tensor(N, C, H, W) with original spatial order
    """
    inv_h = torch.argsort(h_idx)
    inv_w = torch.argsort(w_idx)
    restored = x[:, :, inv_h, :]
    restored = restored[:, :, :, inv_w]
    return restored
