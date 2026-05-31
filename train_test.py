import torch
from torch import nn
from torch.utils.data import DataLoader
from helper_function import accuracy_fn
from typing import Union
from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
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
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Performs a testing loop on model going over data_loader"""
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)
            # 1.forward pass
            test_pred = model(X)
            # 2. Calculate the loss and accuracy (per batch)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Adjust metrics and print out
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")


def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Returns a dictionary contatining the results of model predicting on data_loader"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Device agnostic
            X, y = X.to(device), y.to(device)

            # Make predictions
            y_pred = model(X)
            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            # argmax will function as the softmax activation function as it will return the highest value from the set of output logits
        # Scale the loss and acc to find the average loss / acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        return {
            "model_name": model.__class__.__name__,  # Only works when model is created with a class
            "model_loss": loss.item(),
            "model_acc": acc,
        }
