from model import DeepLabV3Plus
import torch.optim as optim
from visioneerium2.data import load_data
import torch.nn as nn
from train import test_fn, training_loop
from utils import (
    load_checkpoint,
    save_model,
)

# hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 6
NUM_EPOCHS = 6
NUM_WORKERS = 6

# set this to true when you want to load a checkpoint
LOAD_MODEL = False
DEVICE = "cpu"

def main():

    model = DeepLabV3Plus(
        num_classes=4, 
        backbone="resnet50",
        aspp_out_channels=256,
        decoder_channels=256,
        low_level_channels=48,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # get loaders for each set, train, valid, test
    train_loader, val_loader, test_loader = load_data(
        r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\metadata.csv",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    resume_epoch = 0
    if LOAD_MODEL:
        print("continuing...")

        # you wil have to modify this path, and set it to whatever checkpoint you want
        checkpoint_path = "checkpoint_epoch_1.pth.tar" 
        model, optimizer, resume_epoch, _ = load_checkpoint(
            checkpoint_path, model, optimizer
        )

        resume_epoch = resume_epoch or 0 #if load checkpoint fails

        print(f"Resuming from epoch {resume_epoch + 1}...")

    # run the training loop
    training_loop(resume_epoch, train_loader, val_loader, model, optimizer, loss_fn)


    save_model(model, filename="trained_deeplabv3_model.pth")

    test_fn(test_loader, model)


if __name__ == "__main__":
    main()