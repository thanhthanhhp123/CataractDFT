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
from base.net import DualCNN

if not os.path.exists('dualcnn'):
    os.makedirs('dualcnn')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dft_transform(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)
    return magnitude, phase

class CataractFreqDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mode='magnitude'):
        self.root = root
        self.transform = transform
        self.mode = mode

        ds = torchvision.datasets.ImageFolder(root=root, transform=None)
        self.samples = ds.samples
        self.classes = ds.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        magnitude, phase = dft_transform(image)


        image = Image.fromarray((image * 255).astype('uint8'))
        magnitude = Image.fromarray((magnitude * 255).astype('uint8'))
        phase = Image.fromarray((phase * 255).astype('uint8'))
        image_f = Image.fromarray((image_f * 255).astype('uint8'))
        if self.transform:
            image_f = self.transform(image_f)
            image = self.transform(image)
            magnitude = self.transform(magnitude)
            phase = self.transform(phase)
        image = torch.tensor(image, dtype=torch.float32)
        magnitude = torch.tensor(magnitude, dtype=torch.float32)
        phase = torch.tensor(phase, dtype=torch.float32)
        image_f = torch.tensor(image_f, dtype=torch.float32)


        if self.mode == 'magnitude':

            return {'spatial': image_f, 'frequency': magnitude, 'label': label}


        elif self.mode == 'phase':
            return {'spatial': image_f, 'frequency': phase, 'label': label}
        

cataract_freq_train_dataset = CataractFreqDataset(root=r'/content/OculaCare-1/train', transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), mode='magnitude')
cataract_train_loader = DataLoader(cataract_freq_train_dataset, batch_size=32, shuffle=True)

cataract_freq_val_dataset = CataractFreqDataset(root=r'/content/OculaCare-1/valid', transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), mode='magnitude')
cataract_val_loader = DataLoader(cataract_freq_val_dataset, batch_size=32, shuffle=False)

cataract_freq_test_dataset = CataractFreqDataset(root=r'/content/OculaCare-1/test', transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), mode='magnitude')
cataract_test_loader = DataLoader(cataract_freq_val_dataset, batch_size=32, shuffle=False)
freq_model = DualCNN()
freq_model = freq_model.to(device)

num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(freq_model.parameters(), lr=0.001)


train_acc = []
valid_acc = []
losses = []

for epoch in range(num_epochs):
    freq_model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in cataract_train_loader:
        spatial_image = batch['spatial'].to(device)
        freq_image = batch['frequency'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = freq_model(spatial_image, freq_image)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)

    train_loss = running_loss / len(cataract_train_loader)
    train_accuracy = correct_predictions / total_predictions * 100

    losses.append(train_loss)
    train_acc.append(train_accuracy)

    # Validation phase
    freq_model.eval()
    test_correct_predictions = 0
    test_total_predictions = 0

    for batch in cataract_val_loader:
        spatial_image = batch['spatial'].to(device)
        freq_image = batch['frequency'].to(device)
        labels = batch['label'].to(device)

        outputs = freq_model(spatial_image, freq_image)
        _, preds = torch.max(outputs, 1)

        test_correct_predictions += (preds == labels).sum().item()
        test_total_predictions += labels.size(0)

    valid_accuracy = test_correct_predictions / test_total_predictions * 100
    valid_acc.append(valid_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Train Accuracy: {train_accuracy}, Validation Accuracy: {valid_accuracy}')

torch.save(freq_model.state_dict(), 'dualcnn/freq_model.pth')
print('Training completed')

plt.figure(figsize=(10, 7))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.figure(figsize=(10, 7))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('basemodel/loss.png')

plt.figure(figsize=(10, 7))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(valid_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('basemodel/accuracy.png')

#F1 Score, Precision, Recall, confusion matrix and ROC curve testloader

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import pandas as pd

freq_model.eval()
test_correct_predictions = 0
test_total_predictions = 0
y_true = []
y_pred = []

with torch.no_grad():
    for batch in cataract_test_loader:
        spatial_image = batch['spatial'].to(device)
        freq_image = batch['frequency'].to(device)
        labels = batch['label'].to(device)

        y_true.extend(labels.cpu().numpy())

        outputs = freq_model(spatial_image, freq_image)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        test_correct_predictions += (preds == labels).sum().item()
        test_total_predictions += labels.size(0)

test_accuracy = test_correct_predictions / test_total_predictions * 100

f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

print(f"Test Accuracy: {test_accuracy:.2f}%, F1 Score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
print(f"Confusion Matrix: \n{cm}")

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('dualcnn/cm.png')

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('dualcnn/roc.png')

pd.DataFrame({'F1 Score': [f1], 'Precision': [precision], 'Recall': [recall], 'Test Accuracy': [test_accuracy], 'ROC AUC': [roc_auc]}).to_csv('dualcnn/metrics.csv', index=False)

