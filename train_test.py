import torch
from torch import nn
from torch.utils.data import DataLoader
from helper_function import accuracy_fn
from torch import device


def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """peroforms a training with model training to learn on data_loader"""

    model.train()

    train_loss, train_acc = 0, 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(
        train_dataloader
    ):  # Trains the model iteratively on single batches
        X, y = X.to(device), y.to(device)

        # 1.Forward pass
        y_pred = model(X)

        # 2.Calcualte the loss and acccuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(
            y_true=y, y_pred=y_pred.argmax(dim=1)
        )  # y_pred.argmax(dim = 1) returns the highest value from each row at dimension 1

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4.Loss backward
        loss.backward()

        # 5.Optimizer step
        optimizer.step()

        # Print out what's happening
        if batch % 400 == 0:
            print(f"Batch : {batch * len(X)}/{len(train_dataloader.dataset)} ")
    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    print(f"Train loss:{train_loss:.5f}|Train acc:{train_acc:.2f}%")


def test_step(
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Performs a testing loop on model going over data_laoder"""
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)
            # 1.forward pass
            test_pred = model(X)

        test_loss += loss_fn(test_pred, y)
        test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        # Adjust metrics and print out
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
