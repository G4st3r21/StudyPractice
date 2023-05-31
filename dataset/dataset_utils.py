import os
import csv
import random

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import random_split
from dataset import CustomDataset


def augment_image(image_path, num_augmentations):
    image = Image.open(image_path)
    augmented_images = []

    for i in range(num_augmentations):
        augmented_image = image.copy()

        augmented_image = augmented_image.point(lambda p: p * random.uniform(0.5, 1.5))

        augmented_image = augmented_image.convert("HSV")
        augmented_image = augmented_image.point(lambda p: p * random.uniform(0.8, 1.2))
        augmented_image = augmented_image.convert("RGB")

        augmented_image = augmented_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

        augmented_images.append(augmented_image)

    return augmented_images


def prepare_dataset(raw_dataset_folder: str = "../data/raw", augment=True):
    output_dir = '../data/processed/'

    with open('../data/annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])

    folders = os.listdir(raw_dataset_folder)
    for folder in folders:
        images = os.listdir(raw_dataset_folder + "/" + folder)
        for image in images:
            image_path = os.path.join(raw_dataset_folder, folder, image)
            img = cv2.imread(image_path)
            output_path = os.path.join(output_dir, image)
            cv2.imwrite(output_path, img)

            with open('../data/annotations.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([output_path, 1 if folder == "passport" else 0])

            if augment:
                augmented_images = augment_image(image_path, 4)
                for i, augmented_image in enumerate(augmented_images):
                    augmented_image.save(f"{output_dir}/augmented{i}_{image}")

                    with open('../data/annotations.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f"{output_dir}/augmented{i}_{image}", 1 if folder == "passport" else 0])


def prepare_data_loaders(root_dir: str = '../data/processed', annotations="../data/annotations.csv"):
    dataset = CustomDataset(root_dir=root_dir, annotations=annotations, transform=transforms.ToTensor())

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)

    return train_loader, test_loader
