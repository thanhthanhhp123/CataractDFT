import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from base.net import BaseModel
import os
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

if not os.path.exists('/basemodel'):
    os.makedirs('/basemodel')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_ds = torchvision.datasets.ImageFolder(r'/content/OculaCare-1/train', transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean = [0.0096, 0.0076, 0.0067], std = [0.0830, 0.0676, 0.0618])]))
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

valid_ds = torchvision.datasets.ImageFolder(r'/content/OculaCare-1/valid', transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean = [0.0096, 0.0076, 0.0067], std = [0.0830, 0.0676, 0.0618])]))
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=32, shuffle=False)

test_ds = torchvision.datasets.ImageFolder(r'/content/OculaCare-1/test', transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean = [0.0096, 0.0076, 0.0067], std = [0.0830, 0.0676, 0.0618])]))
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)


model = BaseModel(in_channels=3, num_classes=2)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_acc = []
valid_acc = []
losses = []

logging.info('Training started')
for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_dl:
            images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1) 
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        train_loss = running_loss / len(train_dl)
        train_accuracy = correct_predictions / total_predictions * 100  

        losses.append(train_loss)
        train_acc.append(train_accuracy)

        # Validation phase
        model.eval()
        test_correct_predictions = 0
        test_total_predictions = 0

        with torch.no_grad():
            for images, labels in valid_dl:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_correct_predictions += (preds == labels).sum().item()
                test_total_predictions += labels.size(0)

        valid_accuracy = test_correct_predictions / test_total_predictions * 100 

        valid_acc.append(valid_accuracy)
        logging.info(f"Epoch {epoch + 1}/{10}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {valid_accuracy:.2f}%")
        # print(f"Epoch {epoch + 1}/{10}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {valid_accuracy:.2f}%")

torch.save(model.state_dict(), 'basemodel/basemodel.pth')

logging.info('Evaluation started')
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

model.eval()
test_correct_predictions = 0
test_total_predictions = 0
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_dl:
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        y_true.extend(labels.cpu().numpy())

        outputs = model(images)
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
plt.savefig('basemodel/cm.png')

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('basemodel/roc.png')

pd.DataFrame({'F1 Score': [f1], 'Precision': [precision], 'Recall': [recall], 'Test Accuracy': [test_accuracy], 'ROC AUC': [roc_auc]}).to_csv('basemodel/metrics.csv', index=False)








