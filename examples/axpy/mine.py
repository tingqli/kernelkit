import torch


def axpy(x, y, *, alpha=1.0):
    """An experiment's version of axpy: any function of the interface arguments."""
    return torch.addcmul(y, x, torch.full_like(x, alpha))
