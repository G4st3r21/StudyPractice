import csv
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, annotations, transform):
        self.root_dir = root_dir
        self.annotations = annotations
        self.transform = transform

        self.default_images = []
        self.images = []
        self.labels = []
        self.__prepare_images()

    def __prepare_image_size(self):
        for img in self.default_images:
            resized_image = img.resize((1280, 720))
            self.images.append(resized_image)

    def __prepare_images(self):
        with open(self.annotations, 'r') as file:
            csv_annotations = csv.reader(file)
            for item in list(csv_annotations)[1:]:
                file_path = item[0]
                label = item[-1]
                image = Image.open(file_path).convert('RGB')
                self.default_images.append(image)
                self.labels.append(int(label))
        self.__prepare_image_size()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
