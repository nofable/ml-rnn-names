import torch
from torch.nn.utils.rnn import pad_sequence

from ml_rnn_names.NamesDataset import NamesDataset
from ml_rnn_names.utils import get_project_root


def load_data(device):
    project_root = get_project_root()
    alldata = NamesDataset(project_root / "data/names")
    print(f"Loaded {len(alldata)} items of data")
    # Split data into train and test
    train_set, test_set = torch.utils.data.random_split(
        alldata,
        [0.85, 0.15],
        generator=torch.Generator(device=device).manual_seed(2024),
    )
    print(f"Train examples = {len(train_set)}, test examples = {len(test_set)}")
    return train_set, test_set, alldata.labels_uniq


def collate_names(batch):
    # batch is list of (label_tensor, text_tensor, label, text)
    labels = torch.stack([item[0] for item in batch]).squeeze(1)
    texts = [item[1].squeeze(1) for item in batch]  # (seq_len, n_letters) each
    lengths = torch.tensor([t.size(0) for t in texts])

    # Pad to max length in batch: (max_seq_len, batch_size, n_letters)
    padded = pad_sequence(texts, batch_first=False, padding_value=0)
    return labels, padded, lengths
