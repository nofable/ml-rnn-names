from io import open
import glob
import os
import time

import torch
from torch.utils.data import Dataset

from ml_rnn_names.processing import line_to_tensor


class NamesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir  # for provenance of the dataset
        self.load_time = time.localtime  # for provenance of the dataset
        labels_set = set()  # set of all classes { "Chinese", "Czech", ... }

        self.data = []  # a list of all names in all files
        self.data_tensors = []  # same list, but in tensor form
        self.labels = []  # ["Chinese", "Chinese", "Czech" ...]
        self.labels_tensors = []

        # read all the ``.txt`` files in the specified directory
        text_files = glob.glob(os.path.join(data_dir, "*.txt"))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[
                0
            ]  # Irish.txt -> "Irish"
            labels_set.add(label)  # add the label to the set
            lines = open(filename, encoding="utf-8").read().strip().split("\n")
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(line_to_tensor(name))
                self.labels.append(label)

        # Cache the tensor representation of the labels
        self.labels_uniq = list(
            labels_set
        )  # order the set eg. ["Chinese", "Czech"] to give each label an index number

        for label in self.labels:
            temp_tensor = torch.tensor(
                [self.labels_uniq.index(label)], dtype=torch.long
            )
            self.labels_tensors.append(
                temp_tensor
            )  # An index of each label [[0.0], [1.0], [1.0], ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item
