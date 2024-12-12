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
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = CataractFreqDataset(root=r'E:\Project\Cataract_DFT\Dataset\train', transform=transform, mode=None)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    test_ds = CataractFreqDataset(root=r'E:\Project\Cataract_DFT\Dataset\test', transform=transform, mode=None)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False)

    val_ds = CataractFreqDataset(root=r'E:\Project\Cataract_DFT\Dataset\valid', transform=transform, mode=None)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    print(next(iter(train_dl))[0].shape)


    epochs = 10
    model = torchvision.models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, 2)
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_dl):
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch {epoch}, Step {i}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_dl:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Epoch {epoch}, Val Accuracy: {correct/total}')

    
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
    import seaborn as sns
    import pandas as pd

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_dl:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_acc = np.mean(np.array(y_true) == np.array(y_pred))
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f'Test Accuracy: {test_acc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f'ROC AUC: {roc_auc}')
    print(f"Confusion Matrix: \n{cm}")

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('basemodel/cm.png')

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('basemodel/roc.png')

    pd.DataFrame({'F1 Score': [f1], 'Precision': [precision], 'Recall': [recall], 'Test Accuracy': [test_acc], 'ROC AUC': [roc_auc]}).to_csv('basemodel/metrics.csv', index=False)
