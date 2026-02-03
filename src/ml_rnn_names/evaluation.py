import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch import nn
from torch.utils.data import DataLoader
from ml_rnn_names.data import collate_names
from ml_rnn_names.training import evaluate_loss


def compute_accuracy(model, data):
    """Compute accuracy on a dataset."""
    model.eval()
    dataloader = DataLoader(
        data,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_names,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, padded_texts, lengths in dataloader:
            outputs = model(padded_texts, lengths)
            _, predictions = outputs.topk(1)
            predictions = predictions.squeeze(1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0


def compute_confusion_matrix(model, data, n_classes):
    """Compute confusion matrix for a dataset."""
    confusion = torch.zeros(n_classes, n_classes)

    dataloader = DataLoader(
        data,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_names,
    )

    model.eval()
    with torch.no_grad():
        for labels, padded_texts, lengths in dataloader:
            outputs = model(padded_texts, lengths)
            _, predictions = outputs.topk(1)
            predictions = predictions.squeeze(1)

            for label, pred in zip(labels, predictions):
                confusion[label][pred] += 1

    return confusion


def plot_confusion_matrix(confusion, classes, normalize=True):
    """Plot confusion matrix."""
    confusion = confusion.clone()
    if normalize:
        for i in range(len(classes)):
            denom = confusion[i].sum()
            if denom > 0:
                confusion[i] = confusion[i] / denom

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    return fig


def evaluate_model(model, train_data, val_data=None, test_data=None, criterion=None):
    """
    Comprehensive evaluation of a model.

    Returns dict with train/val/test metrics (accuracy and loss).
    """
    if criterion is None:
        criterion = nn.NLLLoss()

    results = {
        "train_accuracy": compute_accuracy(model, train_data),
        "train_loss": evaluate_loss(model, train_data, criterion),
    }

    if val_data is not None:
        results["val_accuracy"] = compute_accuracy(model, val_data)
        results["val_loss"] = evaluate_loss(model, val_data, criterion)

    if test_data is not None:
        results["test_accuracy"] = compute_accuracy(model, test_data)
        results["test_loss"] = evaluate_loss(model, test_data, criterion)

    return results
