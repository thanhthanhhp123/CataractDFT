import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def Trainer(model, optim, train_loader, val_loader, epochs, loss_fn, device='cuda', output_csv="metrics.csv"):
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please set device='cpu'")
    elif device == 'cuda':
        print("Using CUDA device")
    else:
        print("Using CPU device")
    model.to(device)
    metrics_history = []

    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_targets = 0, [], []

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            train_targets.extend(targets.cpu().tolist())

        train_metrics = compute_metrics(train_targets, train_preds)
        train_loss /= len(train_loader)

        model.eval()
        test_loss, test_preds, test_targets = 0, [], []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Testing"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item()
                test_preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                test_targets.extend(targets.cpu().tolist())

        test_metrics = compute_metrics(test_targets, test_preds)
        test_loss /= len(val_loader)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_score": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_score": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
        }
        metrics_history.append(epoch_metrics)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")
    torch.save(model.state_dict(), "model.pth")


def Tester(model, test_loader, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please set device='cpu'")
    elif device == 'cuda':
        print("Using CUDA device")
    else:
        print("Using CPU device")
    model.to(device)
    model.eval()
    test_preds, test_targets = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            test_targets.extend(targets.cpu().tolist())

    test_metrics = compute_metrics(test_targets, test_preds)
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

    pd.DataFrame(test_metrics, index=[0]).to_csv("test_metrics.csv", index=False)
    print("Test metrics saved to test_metrics.csv")
    

