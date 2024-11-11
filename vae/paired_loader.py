# 現在のフレームと次のフレームのペアを返すデータセットクラス
import os
from torchvision import datasets
from PIL import Image


class PairedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.input_dir = os.path.join(root, 'inputs')
        self.target_dir = os.path.join(root, 'targets')
        self.input_images = sorted(os.listdir(self.input_dir))
        self.target_images = sorted(os.listdir(self.target_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.input_images[index])
        target_path = os.path.join(self.target_dir, self.target_images[index])
        
        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

 