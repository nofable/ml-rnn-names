import string
import unicodedata
import torch
import glob
import os
from ml_rnn_names.utils import get_project_root


# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
all_letters = string.ascii_letters + " .,;'_"
n_letters = (
    len(all_letters) + 2
)  # Plus SOS and EOS marker (Start of Sequence, End of Sequence)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if letter not in all_letters:
        return all_letters.find("_")
    else:
        return all_letters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def label_from_output(output, output_labels):
    _top_values, top_indices = output.topk(1)
    label_index = top_indices[0].item()
    return output_labels[label_index], label_index


def findFiles(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding="utf-8") as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]


def createCategoryDict():
    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []
    project_root = get_project_root()
    for filename in findFiles(f"{project_root}/data/names/*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    if n_categories == 0:
        raise RuntimeError(
            "Data not found. Make sure that you downloaded data "
            "from https://download.pytorch.org/tutorial/data.zip and extract it to "
            "the current directory."
        )

    return all_categories, category_lines, n_categories


# One-hot vector for category
def categoryTensor(all_categories, category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, len(all_categories))
    tensor[0][li] = 1
    return tensor


def inputTensor(line=""):
    tensor = torch.zeros(len(line) + 1, 1, n_letters)
    tensor[0][0][n_letters - 2] = 1  # SOS
    for li in range(len(line)):
        letter = line[li]
        tensor[li + 1][0][all_letters.find(letter)] = 1
    return tensor


# ``LongTensor`` of target letters followed by EOS
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)
