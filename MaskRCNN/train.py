# deep learning libraries
import torch

# other libraries
from typing import Final

# own modules
from src.data import load_data
from src.unet.train_functions import train_step, val_step, test_step
from src.utils import set_seed, save_model, parameters_to_double

import os

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    HPP_DICT = dict(
        batch_size=128,
        epochs=30,
        lr=0.001,
        weight_decay=0.0,
        patience=5,
    )

    # Load the data
    train_loader, val_loader, test_loader = load_data(DATA_PATH, HPP_DICT)
    print("Data loaded")

    # Create the model
    model = ...
    parameters_to_double(model)
    print(model)

    # Create the loss function
    criterion = torch.nn.L1Loss()

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HPP_DICT["lr"], weight_decay=HPP_DICT["weight_decay"]
    )

    best_loss = float("inf")
    no_improvement = 0

    # Train the model
    for epoch in range(1, HPP_DICT["epochs"] + 1):
        # Train the model
        train_step(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            device,
        )

        # Evaluate the model
        val_loss = val_step(
            model,
            val_loader,
            criterion,
            device,
        )

        print("------------\nEPOCH: ", epoch)
        print("Validation Loss: ", val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            # Save the model
            i = len(os.listdir("models")) + 1
            save_model(model, f"model_{i}")

            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > HPP_DICT["patience"]:
                break

    # Test the model
    i = len(os.listdir("models"))
    best_model = torch.jit.load(f"models/model_{i}.pt")
    test_loss = test_step(best_model, test_loader, device, criterion)
    print("Test Loss: ", test_loss)


if __name__ == "__main__":
    main()

