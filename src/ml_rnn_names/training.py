import random
import numpy as np

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from ml_rnn_names.data import collate_names


def train(
    model,
    training_data,
    n_epoch=10,
    learning_rate=0.2,
    n_batch_size=64,
    report_every=10,
    criterion=nn.NLLLoss(),
    device=None,
):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    # current_loss = 0
    all_losses = []
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    dataloader = DataLoader(
        training_data,
        batch_size=n_batch_size,
        shuffle=True,
        collate_fn=collate_names,
        #   pin_memory=True,  # Faster CPU->GPU transfer
        generator=torch.Generator(device=device).manual_seed(2024),
    )

    start = time.time()
    print(f"Training on data set with n = {len(training_data)}")

    for epoch in range(1, n_epoch + 1):
        epoch_loss = 0
        n_batches = 0

        for labels, padded_texts, lengths in dataloader:
            optimizer.zero_grad()
            output = model(padded_texts, lengths)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 3
            )  # cap gradients to 3 to avoid large jumps
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        all_losses.append(avg_loss)

        if epoch % report_every == 0:
            print(
                f"{epoch} ({epoch / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}"
            )

    end = time.time()
    print(f"Training took {end - start}s")
    print(f"Final average loss/batch = {all_losses[-1]}")

    return all_losses
