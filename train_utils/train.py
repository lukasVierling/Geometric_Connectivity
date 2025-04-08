import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def train_epoch(model, optimizer, criterion, train_loader, device, epoch):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    pbar = tqdm(train_loader, desc=f"Training in epoch {epoch}...")
    for batch_idx, (x,y) in enumerate(pbar):
        #get bs
        batch_size = x.shape[0]
        #zero grad
        optimizer.zero_grad()

        #forward and loss calc
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out,y)

        #optimizer step
        loss.backward()
        optimizer.step()

        #track loss
        total_loss += loss.item() * batch_size # * batch_size

        #get correct predictions
        _, predicted = out.max(1)
        batch_correct = predicted.eq(y).sum().item()
        acc = batch_correct/batch_size
        total += batch_size
        correct += batch_correct

        pbar.set_description(f"Current loss: {loss} on batch: {batch_idx} with acc: {acc}")
        
        #weights and biases
        wandb.log({ "epoch": epoch,
                    "batch_idx":batch_idx,
                    "train_loss": loss.item(),
                    "train_acc": acc})



