import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvolutionalExploreModel(nn.Module):
    """
    This applies right, left, and forward gradient filters to the gridmap scan.
    Each result is then summed to get a single number for each direction.

    NOTE: This method assumes that the scan is interpolated to be facing North.
    """

    def __init__(self, filter_size: tuple = (10, 10)):
        super(ConvolutionalExploreModel, self).__init__()
        self.filter_size = filter_size
        self.forward_filter = nn.Parameter(torch.zeros(filter_size))
        self.left_filter = nn.Parameter(torch.zeros(filter_size))
        self.right_filter = nn.Parameter(torch.zeros(filter_size))

    def forward(self, scan: torch.Tensor):
        if scan.shape != self.filter_size:
            raise ValueError(
                f"Gridmap scan tensor must match filter tensor size. Expected {self.filter_size}, but got {scan.shape}"
            )

        fwd_matrix = self.forward_filter * scan
        left_matrix = self.left_filter * scan
        right_matrix = self.right_filter * scan

        fwd = torch.sum(fwd_matrix).unsqueeze(0)
        left = torch.sum(left_matrix).unsqueeze(0)
        right = torch.sum(right_matrix).unsqueeze(0)

        out = torch.cat([fwd, left, right], dim=0)

        return F.softmax(out, dim=0)

    def set_filters(
        self, forward: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ):
        if (
            right.shape != self.filter_size
            or left.shape != self.filter_size
            or forward.shape != self.filter_size
        ):
            raise ValueError(
                f"Filter tensors must match filter size. Expected {self.filter_size}, but got {right.shape}, {left.shape}, {forward.shape}"
            )

        self.right_filter.data = right
        self.left_filter.data = left
        self.forward_filter.data = forward

    def get_filters(self):
        return self.forward_filter.data, self.left_filter.data, self.right_filter.data

    def add_to_filters(
        self, forward: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ):
        if (
            right.shape != self.filter_size
            or left.shape != self.filter_size
            or forward.shape != self.filter_size
        ):
            raise ValueError(
                f"Filter arrays must match filter size. Expected {self.filter_size}, but got {right.shape}, {left.shape}, {forward.shape}"
            )

        self.right_filter.data += right
        self.left_filter.data += left
        self.forward_filter.data += forward

    def multiply_filters(
        self, forward: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ):
        if (
            right.shape != self.filter_size
            or left.shape != self.filter_size
            or forward.shape != self.filter_size
        ):
            raise ValueError(
                f"Filter arrays must match filter size. Expected {self.filter_size}, but got {forward.shape}, {left.shape}, {right.shape}"
            )

        self.right_filter.data *= right
        self.left_filter.data *= left
        self.forward_filter.data *= forward


def create_gradient_filters(filter_size):
    rows, cols = filter_size
    mid_row, mid_col = rows // 2, cols // 2

    # Left gradient filter
    left_filter = np.zeros((rows, cols))
    for i in range(mid_row + 1):
        left_filter[i, :mid_col] = np.linspace(1, 0, mid_col)

    # Right gradient filter
    right_filter = np.zeros((rows, cols))
    for i in range(mid_row + 1):
        right_filter[i, mid_col:] = np.linspace(0, 1, mid_col)

    # Forward gradient filter
    forward_filter = np.zeros((rows, cols))
    start = mid_col // 2
    end = mid_col + (mid_col // 2) + 1
    for i in range(mid_row + 1):
        forward_filter[i, mid_col:end] = np.linspace(0.9, 0, end - mid_col)
        forward_filter[i, start:mid_col] = np.linspace(
            0,
            0.8,
            mid_col - start,
        )

    return (
        torch.tensor(forward_filter, dtype=torch.float32),
        torch.tensor(left_filter, dtype=torch.float32),
        torch.tensor(right_filter, dtype=torch.float32),
    )


# Example usage
model = ConvolutionalExploreModel()

filter_size = (10, 10)
forward_filter, left_filter, right_filter = create_gradient_filters(filter_size)
model.set_filters(forward_filter, left_filter, right_filter)

# Get filters
fwd, left, right = model.get_filters()
print("Filters:\n\n", "forward:\n", fwd, "\nleft:\n", left, "\nright:\n", right)

# Example scan
scan = torch.rand(10, 10)
output = model.forward(scan)
print("\n\nOutput:", output)

# Add to filters
model.add_to_filters(forward_filter, left_filter, right_filter)
# print("Filters after addition:", model.get_filters())

# Multiply filters
model.multiply_filters(forward_filter, left_filter, right_filter)
# print("Filters after multiplication:", model.get_filters())
