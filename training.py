from src.trainer import Trainer
from src.utils import CataractFreqDataset

import argparse
import torch
import torchvision

def main(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    train_dataset = CataractFreqDataset(root=args.train_dir, transform=transform, mode=args.mode)
    val_dataset = CataractFreqDataset(root=args.val_dir, transform=transform, mode=args.mode)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = torchvision.models.resnet18(weights = None)
    
    if args.mode == 'concat':
        model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(512, len(train_dataset.classes))
    elif args.mode == 'magnitude' or args.mode == 'base':
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(512, len(train_dataset.classes))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    Trainer(model, optim, train_loader, val_loader, args.epochs, loss_fn, device=args.device, output_csv=args.output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Path to training dataset", default=r"Dataset\train")
    parser.add_argument("--val_dir", type=str, help="Path to validation dataset", default=r"Dataset\valid")
    parser.add_argument("--mode", type=str, default="base", help="Mode to use for dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--output_csv", type=str, default="metrics.csv", help="Output CSV file for metrics")
    args = parser.parse_args()

    main(args)
