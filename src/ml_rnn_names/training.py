import random
import numpy as np

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from ml_rnn_names.data import collate_names


def set_seed(seed, device=None):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def evaluate_loss(model, data, criterion=nn.NLLLoss()):
    """Compute average loss on a dataset."""
    model.eval()
    dataloader = DataLoader(
        data,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_names,
    )
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for labels, padded_texts, lengths in dataloader:
            output = model(padded_texts, lengths)
            loss = criterion(output, labels)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0


def train(
    model,
    training_data,
    validation_data=None,
    n_epoch=10,
    learning_rate=0.2,
    n_batch_size=64,
    report_every=10,
    criterion=nn.NLLLoss(),
    device=None,
    seed=2024,
):
    """
    Train model on training_data, optionally evaluating on validation_data.

    Returns:
        dict with keys:
            - "train_losses": list of average training loss per epoch
            - "val_losses": list of validation loss per epoch (None if no validation_data)
    """
    set_seed(seed, device)

    all_train_losses = []
    all_val_losses = [] if validation_data is not None else None

    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    dataloader = DataLoader(
        training_data,
        batch_size=n_batch_size,
        shuffle=True,
        collate_fn=collate_names,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    start = time.time()
    print(f"Training on data set with n = {len(training_data)}")

    for epoch in range(1, n_epoch + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for labels, padded_texts, lengths in dataloader:
            optimizer.zero_grad()
            output = model(padded_texts, lengths)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        all_train_losses.append(avg_train_loss)

        if validation_data is not None:
            val_loss = evaluate_loss(model, validation_data, criterion)
            all_val_losses.append(val_loss)

        if epoch % report_every == 0:
            msg = f"{epoch} ({epoch / n_epoch:.0%}): train_loss={avg_train_loss:.4f}"
            if validation_data is not None:
                msg += f", val_loss={all_val_losses[-1]:.4f}"
            print(msg)

    end = time.time()
    print(f"Training took {end - start:.2f}s")

    return {
        "train_losses": all_train_losses,
        "val_losses": all_val_losses,
    }
