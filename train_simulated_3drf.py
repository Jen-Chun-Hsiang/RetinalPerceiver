import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Importing modules from your project
from data.dataset import YourDataset
from models.model import YourModel
from utils.trainer import train_one_epoch, evaluate
from utils.utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")

    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.pth', help='Path to save the model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Flag to load the model from checkpoint')

    return parser.parse_args()

def main():
    args = parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloader
    train_dataset = YourDataset(root='./data', train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = YourDataset(root='./data', train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, loss and optimizer
    model = YourModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Optionally, load from checkpoint
    if args.load_checkpoint:
        start_epoch, model, optimizer = load_checkpoint(args.checkpoint_path, model, optimizer, device)
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)
        evaluate(test_loader, model, device)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, args.checkpoint_path)

if __name__ == '__main__':
    main()
