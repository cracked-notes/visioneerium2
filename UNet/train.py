import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from original_unet import UNet
from src.unet.data import load_data
from src.unet.utils import (
    load_checkpoint,
    save_checkpoint,
    calculate_val_loss,
    save_model,
)

# Hyperparameters
LEARNING_RATE = 0.001
DEVICE = "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 1
NUM_WORKERS = 6
PIN_MEMORY = False
LOAD_MODEL = True
CHECKPOINT_PATH = "checkpoint_epoch_4.pth.tar"


# Function to train the model
def train_fn(
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        loader: DataLoader for the training set.
        model: Model to be trained.
        optimizer: Optimizer for the model.
        loss_fn: Loss function for the model.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    train_loss = 0.0

    loop = tqdm(loader, total=len(loader), desc="Training", leave=True)
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Forward
        predictions = model(images)
        loss = loss_fn(predictions, masks.long())
        train_loss += loss.item() * images.size(0)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar with loss information
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(loader.dataset)

    return avg_train_loss


def calculate_iou(
    prediction: torch.Tensor, target: torch.Tensor, class_value: int
) -> float:
    """
    Calculate Intersection over Union for a specific class.

    Args:
        prediction: Predicted mask.
        target: Target mask.
        class_value: Value of the class for which IoU is to be calculated.

    Returns:
        float: Intersection over Union for the class.
    """
    # Convert to a binary mask for the specific class
    pred_class = (prediction == class_value).float()
    target_class = (target == class_value).float()

    # Calculate IoU
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection
    if union == 0:
        return float("nan")  # Avoid division by zero if both are empty
    else:
        return intersection / union


def test_fn(loader: DataLoader, model: nn.Module, device: str = "cuda"):
    """
    Evaluate the model on the test set.

    Args:
        loader: DataLoader for the test set.
        model: Model to be evaluated.
        device: Device to run the model on. Default: "cuda".
    """
    model.eval()

    head_ious = []
    flippers_ious = []
    carapace_ious = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            predictions = torch.argmax(torch.sigmoid(outputs), dim=1)

            # Calculate IoU for each category
            carapace_ious.append(
                torch.tensor(calculate_iou(predictions, targets, class_value=1))
            )
            flippers_ious.append(
                torch.tensor(calculate_iou(predictions, targets, class_value=2))
            )
            head_ious.append(
                torch.tensor(calculate_iou(predictions, targets, class_value=3))
            )

        # Filter out NaN values
        head_ious = [iou for iou in head_ious if not torch.isnan(iou)]
        flippers_ious = [iou for iou in flippers_ious if not torch.isnan(iou)]
        carapace_ious = [iou for iou in carapace_ious if not torch.isnan(iou)]

        # Calculate mean IoU for each category
        mean_head_iou = torch.mean(torch.stack(head_ious))
        mean_flippers_iou = torch.mean(torch.stack(flippers_ious))
        mean_carapace_iou = torch.mean(torch.stack(carapace_ious))

        print(f"Head mIoU: {mean_head_iou:.4f}")
        print(f"Flippers mIoU: {mean_flippers_iou:.4f}")
        print(f"Carapace mIoU: {mean_carapace_iou:.4f}")


def plot_loss_graph(train_losses: list):
    """
    Plots the training loss graph.

    Args:
        train_losses: List of training losses for each epoch
    """
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.title("Training Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def train(path: str = "data/"):
    """
    Main training function to train the model using the dataset.

    Args:
        path: Path to the dataset. Default: "data/".
    """
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = load_data(
        path=path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    loss_list = []

    if LOAD_MODEL:  # Resume training from a checkpoint
        print("continuing...")

        # Load checkpoint
        model, optimizer, start_epoch, _ = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, device=DEVICE
        )
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if start_epoch is None:
            start_epoch = 0

        print(f"Resuming from epoch {start_epoch + 1} / {NUM_EPOCHS}...")

        # Training loop
        for epoch in range(start_epoch + 1, NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

            avg_loss = train_fn(train_loader, model, optimizer, loss_fn)
            loss_list.append(avg_loss)

            save_checkpoint(
                model,
                optimizer,
                epoch,
                filename=f"checkpoint_epoch_{epoch + 1}.pth.tar",
            )

            calculate_val_loss(val_loader, model, device=DEVICE)

    else:  # Start training from scratch

        # Training loop
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

            avg_loss = train_fn(train_loader, model, optimizer, loss_fn)
            loss_list.append(avg_loss)

            save_checkpoint(
                model, optimizer, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar"
            )

            calculate_val_loss(val_loader, model, device=DEVICE)

    save_model(model, filename="trained_model.pth")

    test_fn(test_loader, model, device=DEVICE)


if __name__ == "__main__":
    train(r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data")
