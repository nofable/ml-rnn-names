import pytest
import torch

from ml_rnn_names.data import collate_names
from ml_rnn_names.processing import n_letters


def test_collate_names_shapes():
    """Test collate_names with explicit input/output shapes.

    Input batch structure (from NamesDataset.__getitem__):
        Each item is a tuple: (label_tensor, text_tensor, label_str, text_str)
        - label_tensor: shape (1,) - class index
        - text_tensor: shape (seq_len, 1, n_letters) - one-hot encoded characters

    Output structure:
        - labels: shape (batch_size,) - stacked class indices
        - padded: shape (max_seq_len, batch_size, n_letters) - padded one-hot sequences
        - lengths: shape (batch_size,) - original sequence lengths
    """
    batch_size = 3
    seq_lens = [4, 2, 5]  # "John", "Li", "Maria"
    max_seq_len = max(seq_lens)

    # Create mock batch items matching NamesDataset output format
    batch = []
    for i, seq_len in enumerate(seq_lens):
        label_tensor = torch.tensor([i])  # shape: (1,)
        text_tensor = torch.rand(seq_len, 1, n_letters)  # shape: (seq_len, 1, n_letters)
        batch.append((label_tensor, text_tensor, f"label_{i}", f"name_{i}"))

    # Call collate_names
    labels, padded, lengths = collate_names(batch)

    # Verify output shapes
    assert labels.shape == (batch_size,), f"Expected labels shape ({batch_size},), got {labels.shape}"
    assert padded.shape == (max_seq_len, batch_size, n_letters), (
        f"Expected padded shape ({max_seq_len}, {batch_size}, {n_letters}), got {padded.shape}"
    )
    assert lengths.shape == (batch_size,), f"Expected lengths shape ({batch_size},), got {lengths.shape}"

    # Verify lengths match input sequence lengths
    assert lengths.tolist() == seq_lens

    # Verify labels are correctly extracted
    assert labels.tolist() == [0, 1, 2]


def test_collate_names_padding_value():
    """Test that padding uses -1 as the padding value."""
    # Create two sequences of different lengths
    batch = [
        (torch.tensor([0]), torch.ones(3, 1, n_letters), "label_0", "abc"),
        (torch.tensor([1]), torch.ones(1, 1, n_letters), "label_1", "x"),
    ]

    _labels, padded, _lengths = collate_names(batch)

    # First sequence (length 3) should have no padding
    assert (padded[:, 0, :] != -1).all()

    # Second sequence (length 1) should be padded at positions 1 and 2
    assert (padded[0, 1, :] == 1).all()  # Original data (ones)
    assert (padded[1, 1, :] == -1).all()  # Padded
    assert (padded[2, 1, :] == -1).all()  # Padded
