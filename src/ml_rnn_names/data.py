import torch
from torch.nn.utils.rnn import pad_sequence

from ml_rnn_names.NamesDataset import NamesDataset
from ml_rnn_names.utils import get_project_root


def load_data(device, validation_split=0.0, seed=2024):
    """
    Load the names dataset with train/validation/test splits.

    Args:
        device: torch device
        validation_split: fraction for validation (default 0.0 for backward compat)
        seed: random seed for reproducibility

    Returns:
        train_set, val_set (None if validation_split=0), test_set, labels_uniq
    """
    project_root = get_project_root()
    alldata = NamesDataset(project_root / "data/names")
    print(f"Loaded {len(alldata)} items of data")

    test_split = 0.15
    train_split = 1.0 - test_split - validation_split
    generator = torch.Generator(device=device).manual_seed(seed)

    if validation_split > 0.0:
        train_set, val_set, test_set = torch.utils.data.random_split(
            alldata,
            [train_split, validation_split, test_split],
            generator=generator,
        )
        print(f"Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        return train_set, val_set, test_set, alldata.labels_uniq
    else:
        train_set, test_set = torch.utils.data.random_split(
            alldata,
            [train_split, test_split],
            generator=generator,
        )
        print(f"Train={len(train_set)}, Test={len(test_set)}")
        return train_set, None, test_set, alldata.labels_uniq


def collate_names(batch):
    # batch is list of (label_tensor, text_tensor, label, text)
    labels = torch.stack([item[0] for item in batch]).squeeze(1)
    texts = [item[1].squeeze(1) for item in batch]  # (seq_len, n_letters) each
    lengths = torch.tensor([t.size(0) for t in texts])

    # Pad to max length in batch: (max_seq_len, batch_size, n_letters)
    padded = pad_sequence(texts, batch_first=False, padding_value=-1)
    return labels, padded, lengths
