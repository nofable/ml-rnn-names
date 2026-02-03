import pytest
import torch

from ml_rnn_names.processing import (
    unicode_to_ascii,
    letter_to_index,
    line_to_tensor,
    label_from_output,
)


def test_unicode_to_ascii():
    assert unicode_to_ascii("w!@~`8786fg") == "wfg"
    assert unicode_to_ascii("0123456789;,.") == ";,."


def test_letter_to_index():
    assert letter_to_index("a") == 0
    assert letter_to_index(".") == 53


def test_line_to_tensor():
    t = line_to_tensor("aaaa")
    idx = letter_to_index("a")
    assert torch.all(t[:, 0, idx] == 1)
    assert torch.sum(t) == 4


def test_label_from_output():
    output = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    output_labels = ["red", "green", "white", "blue", "yellow"]
    assert label_from_output(output, output_labels) == ("red", 0)
