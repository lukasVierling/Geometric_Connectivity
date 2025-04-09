import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

#my imports 
from models.CNN import CNN


def evaluate_loss(model, train_loader,test_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in tqdm(train_loader):
            x,y  = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            total_loss += loss.item()
        for x,y in tqdm(test_loader):
            x,y  = x.to(device), y.to(device)
            out = model(x)
            batch_size = x.shape[0]
            _, predicted = out.max(1)
            batch_correct = predicted.eq(y).sum().item()
            total += batch_size
            correct += batch_correct

    return total_loss / len(train_loader), correct/total

def linear_interpolated_model(model_1, model_2, alpha):
    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()

    state_dict_3 = {}
    for key in state_dict_1.keys():
        state_dict_3[key] = state_dict_1[key] * alpha + (1 - alpha) * state_dict_2[key]

    interpolated_model = CNN()
    interpolated_model.load_state_dict(state_dict_3)
    return interpolated_model
    