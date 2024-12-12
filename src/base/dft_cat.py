import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

from utils import FrequencyPatchMasking

def dft_transform(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)
    return magnitude, phase

class CataractFreqDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mode='concat'):
        self.root = root
        self.transform = transform
        self.mode = mode

        self.fpm = FrequencyPatchMasking(mask_size=(10, 10))

        ds = torchvision.datasets.ImageFolder(root=root, transform=None)
        self.samples = ds.samples
        self.classes = ds.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        augmented_image = self.fpm.process(image)
        if self.transform:
            augmented_image = Image.fromarray(augmented_image)
            augmented_image = self.transform(augmented_image)
            image = Image.fromarray(image)
            image = self.transform(image)

        if self.mode == 'concat':
            return torch.cat((image, augmented_image), dim=0), label
        elif self.mode == 'dft':
            magnitude, phase = dft_transform(image)
            return image, phase, label
        else:
            return augmented_image, label
    
if __name__ == '_main_':
    import matplotlib.pyplot as plt
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CataractFreqDataset(root='/Users/tranthanh/Documents/Projects/CataractDFT/OculaCare.v1i.folder/train', transform=transform, mode='concat')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
        print(images[0][0].shape)
        print(labels)
        image = images[0][0].permute(1, 0).numpy()
        augmented_image = images[0][1].permute(1, 0).numpy()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Augmented Image")
        plt.imshow(augmented_image, cmap='gray')
        plt.show()
        break