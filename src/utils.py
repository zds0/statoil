"""Training utilities."""

import numpy as np

import torch


class MetricMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        """Reset the meter if needed."""
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter with new value."""
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(y_true, y_pred):
    """Accuracy calc."""
    # y_true = y_true.float()
    # _, y_pred = torch.max(y_pred, dim=-1)
    # return (y_pred.float() == y_true).float().mean()
    return (np.round(y_pred) == y_true).float().mean()
