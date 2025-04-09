import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def evaluate(model, test_loader, device, epoch):
    model.eval()
    acc = 0
    pbar = tqdm(test_loader, desc="Testing on Test set...")
    total = 0
    correct = 0
    with torch.no_grad():
        for x,y in pbar:
            batch_size = x.shape[0]
            x,y = x.to(device), y.to(device)
            #get outputs
            out = model(x)
            #get correct predictions
            _, predicted = out.max(1)
            batch_correct = predicted.eq(y).sum().item()
            acc = batch_correct/batch_size
            total += batch_size
            correct += batch_correct

            pbar.set_description(f"Current accuracy: {acc}")
    #weights and biases
    acc = correct / total
    wandb.log({ "epoch": epoch,
                "test_acc": acc})
