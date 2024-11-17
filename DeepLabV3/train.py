import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
# import segmentation_models_pytorch as smp
from model import DeepLabV3Plus
from data import load_data
import matplotlib.pyplot as plt
from utils import (
    load_checkpoint,
    save_checkpoint,
    calculate_val_loss,
    save_model
)

# Hyperparameters
LEARNING_RATE = 0.001
DEVICE = "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 6
NUM_WORKERS = 6
PIN_MEMORY = False
LOAD_MODEL = False

# Lists to store loss values
train_losses = []

# Function to train the model
def train_fn(loader, model, optimizer, loss_fn):
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
        optimizer.zero_grad()  ##
        loss.backward()
        optimizer.step()

        # Update progress bar with loss information
        loop.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss / len(loader.dataset)
    train_losses.append(avg_train_loss)


#####################################
########TESTING###############
def calculate_iou(prediction, target, class_value):
    # Convert the mask for the specific class
    pred_class = (prediction == class_value).float()
    target_class = (target == class_value).float()

    # Calculate IoU
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection
    if union == 0:
        return float("nan")  # Avoid division by zero if both are empty
    else:
        return intersection / union


def test_fn(loader, model):
    model.eval()
    head_ious = []
    flippers_ious = []
    carapace_ious = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            outputs = model(data)
            predictions = torch.argmax(torch.sigmoid(outputs), dim=1)
            predictions_np = predictions.cpu().detach().numpy()

            # Calculate IoU for each category
            carapace_ious.append(torch.tensor(calculate_iou(predictions, targets, class_value=1)))
            flippers_ious.append(torch.tensor(calculate_iou(predictions, targets, class_value=2)))
            head_ious.append(torch.tensor(calculate_iou(predictions, targets, class_value=3)))
            #plt.figure(figsize=(10, 5))

            # # Prediction
            # plt.subplot(1, 2, 1)
            # plt.imshow(predictions_np[0])  # Adjust colormap if necessary
            # plt.title('Prediction : our model')
            # plt.axis('off')

            # # Ground truth
            # plt.subplot(1, 2, 2)
            # plt.imshow(targets[0])  # Adjust colormap if necessary
            # plt.title('Ground Truth : expected image')
            # plt.axis('off')

            # plt.show()

            # break
        # Filter out NaN values and compute mean IoU for each category over the test set
        head_ious = [iou for iou in head_ious if not torch.isnan(iou)]
        flippers_ious = [iou for iou in flippers_ious if not torch.isnan(iou)]
        carapace_ious = [iou for iou in carapace_ious if not torch.isnan(iou)]

        mean_head_iou = torch.mean(torch.stack(head_ious))  # Use torch.stack to concatenate the tensors
        mean_flippers_iou = torch.mean(torch.stack(flippers_ious))  # Use torch.stack to concatenate the tensors
        mean_carapace_iou = torch.mean(torch.stack(carapace_ious))  # Use torch.stack to concatenate the tensors

        print(f"Head mIoU: {mean_head_iou:.4f}")
        print(f"Flippers mIoU: {mean_flippers_iou:.4f}")
        print(f"Carapace mIoU: {mean_carapace_iou:.4f}")
    
    model.train()

###############################################

def plot_loss_graph(train_losses):
    epochs = range(1, len(train_losses) + 1)  # x-axis: epoch numbers
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.title("Training Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    # Initialize pre-built UNET model (from segmentation_models_pytorch)
    model = DeepLabV3Plus(
        num_classes=4,  # 4 classes: background, head, flippers, carapace
        backbone="resnet50",
        aspp_out_channels=256,
        decoder_channels=256,
        low_level_channels=48,
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Get data loaders
    train_loader, val_loader, test_loader = load_data(
        r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\images",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    if LOAD_MODEL:
        print("continuing...")
        checkpoint_path = "checkpoint_epoch_1.pth.tar"  # Path to your checkpoint
        model, optimizer, start_epoch, _ = load_checkpoint(
            checkpoint_path, model, optimizer
        )

        if start_epoch is None:  # Ensure start_epoch is not None
            start_epoch = 0  # Default to epoch 0 if None

        print(f"Resuming from epoch {start_epoch + 1}...")

         # Training loop
        for epoch in range(start_epoch + 1, NUM_EPOCHS + 1): 
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

            train_fn(train_loader, model, optimizer, loss_fn)

            save_checkpoint(
                model, optimizer, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar"
            )

            # Check accuracy on validation set
            calculate_val_loss(val_loader, model)
        
        # Plot the training loss graph
        plot_loss_graph(train_losses)

    else:

        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

            train_fn(train_loader, model, optimizer, loss_fn)

            save_checkpoint(
                model, optimizer, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar"
            )

            # Check accuracy on validation set
            calculate_val_loss(val_loader, model)
        
        # Plot the training loss graph
        plot_loss_graph(train_losses)

     # Save the final trained model
    save_model(model, filename="trained_deeplabv3_model.pth")

    test_fn(test_loader, model)


if __name__ == "__main__":
    main()