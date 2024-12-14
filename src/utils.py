import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision.datasets import ImageFolder
from PIL import Image

def dft_transform(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)
    return magnitude, phase

class CataractFreqDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mode='base', dft_transform=dft_transform):
        self.root = root
        self.transform = transform
        self.mode = mode
        if mode not in ['base', 'concat', 'magnitude']:
            raise ValueError("Invalid mode. Choose from 'base', 'concat', 'magnitude'")

        self.dft = dft_transform

        ds = ImageFolder(root=root, transform=None)
        self.samples = ds.samples
        self.classes = ds.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        magnitude, phase = dft_transform(image)
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
            magnitude = Image.fromarray(magnitude)
            magnitude = self.transform(magnitude)

        if self.mode == 'concat':
            return torch.cat((image, magnitude), dim=0), label
        elif self.mode == 'base':
            return image, label
        elif self.mode == 'magnitude':
            return magnitude, label
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToTensor

    dataset = CataractFreqDataset(root=r"E:\Project\Cataract_DFT\Dataset\train", transform=ToTensor(), mode='concat')
    image, label = dataset[0]
    print(image.shape)