import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])


        input_image = Image.open(input_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return {"input": input_image, "target": target_image}

def get_dataloader(input_dir, target_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImagePairDataset(input_dir=input_dir, target_dir=target_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)