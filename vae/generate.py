# 学習済みのVAEモデルからtest用画像について次のフレームを生成
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model_vae import VAE
from paired_loader import PairedImageFolder

os.makedirs('model', exist_ok=True)
os.makedirs('results', exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((320, 240)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

inverse_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    transforms.Lambda(lambda t: (t * 255).byte()),
    transforms.ToPILImage(),
    transforms.Resize((320, 240)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
checkpoint = torch.load('model/vae.pth.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

test_dataset = PairedImageFolder(root='../data/dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

os.makedirs('generated_images', exist_ok=True)

def test():
    model.eval()

    with torch.no_grad():
        test_loss = 0
        for i, (data, target_data) in enumerate(test_loader):
            print(f"Processing image {i}")
            data = data.to(device)
            
            recon, _, _ = model(data)
            recon_image = recon.cpu()
            inversed_recon_image = inverse_transform(recon_image)
            
            inversed_recon_image.save(f'generated_images/genimg_{i}.png')
        
        print(f"Test loss: {test_loss / len(test_loader)}")


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    test()
