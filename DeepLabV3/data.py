import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import pycocotools.mask as cocoMask
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import logging
# Suppress pycocotools logging
logging.getLogger("pycocotools").setLevel(logging.ERROR)
coco = COCO(r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\annotations.json")

class TurtleDataset(Dataset):

    def __init__(self, split_type: str, path: str) -> None:
        self.path = path
        self.names = os.listdir(path)
        self.split_type = split_type

        metadata = pd.read_csv(r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\metadata_splits.csv")
        self.img_ids = metadata[metadata["split_open"] == split_type]["id"].tolist()
        self.max_width, self.max_height = self.find_max_dimensions()

    def generate_mask(self, img_id: int, img: np.ndarray) -> np.ndarray:
        img_info = coco.loadImgs(img_id)[0]
        img_width = img_info["width"]
        img_height = img_info["height"]
        mask_head = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_carapace = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_flippers = np.zeros((img_height, img_width), dtype=np.uint8)

        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        for ann in anns:
            cat_id = ann["category_id"]
            rle = ann["segmentation"]
            segmentation = ann["segmentation"]

            # Check if segmentation is in uncompressed RLE format
            if (
                isinstance(segmentation, dict)
                and "counts" in segmentation
                and isinstance(segmentation["counts"], list)
            ):
                # Convert uncompressed RLE to compressed RLE
                rle = cocoMask.frPyObjects([segmentation], *segmentation["size"])[0]
            else:
                rle = segmentation  # Already in compressed format

            # Decode the segmentation mask (RLE encoded)
            rle_mask = cocoMask.decode(rle)
            rle_mask = np.array(Image.fromarray(rle_mask))

            # Assign each category to its respective mask
            if cat_id == 1:  # Turtle body
                mask_carapace[rle_mask == 1] = 1
            elif cat_id == 2:  # Flippers
                mask_flippers[rle_mask == 1] = 2
            elif cat_id == 3:  # Head
                mask_head[rle_mask == 1] = 3

        # Combine masks: head and flippers take precedence over the turtle body
        mask = np.maximum(mask_carapace, mask_flippers)  # Ensure flippers overlay body
        mask = np.maximum( mask, mask_head)  # Ensure head overlays both body and flippers
        return mask

    def display_img_and_mask(self, img_id):
        img = coco.imgs[img_id]
        image = np.array(Image.open(img["file_name"]))

        # image
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])

    def find_max_dimensions(self):
        max_width, max_height = 0, 0
        for img_id in self.img_ids:
            img_info = coco.imgs[img_id]
            width, height = img_info["width"], img_info["height"]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        return max_width, max_height

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_id = self.img_ids[index]
        image_info = coco.loadImgs(img_id)

        file_name = image_info[0]['file_name']
        image_path = rf"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\{file_name}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate the combined mask
        mask = self.generate_mask(img_id, image)

        # Padding
        pad_height = self.max_height - image.shape[0]
        pad_width = self.max_width - image.shape[1]
        image = cv2.copyMakeBorder(
            image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        padded_mask = cv2.copyMakeBorder(
            mask, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0
        )

        # Resize image and mask
        target_size = (256, 256)
        image = cv2.resize(image, target_size)
        padded_mask = cv2.resize(padded_mask, target_size)

        # Apply transformations to image
        transformations = transforms.ToTensor()  # Convert to Tensor for image
        image = transformations(image)

        # Convert the mask to tensor
        padded_mask = torch.tensor(padded_mask, dtype=torch.uint8)

        return image, padded_mask


def load_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns dataloaders for training, validation, and test sets.

    Args:
        path: path of the dataset.
        batch_size: batch size for dataloaders. Default value: 128.
        num_workers: number of workers for loading data. Default value: 0.

    Returns:
        tuple of dataloaders, train, val, and test in respective order.
    """

    # create datasets
    train_dataset = TurtleDataset("train", path)
    val_dataset = TurtleDataset("valid", path)
    test_dataset = TurtleDataset("test", path)

    # train_dataset.generate_mask(1, I)

    # define dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data(r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\images")