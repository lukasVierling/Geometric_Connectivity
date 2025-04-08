import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import json

#my imports
from models.CNN import CNN

def save_model_and_config(model, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_data_loaders(dataset_config):
    dataset_name = dataset_config["dataset"]
    batch_size = dataset_config["batch_size"]
    data_root = dataset_config.get("data_root", "./data")
    data_augmentation = dataset_config["data_augmentation"]
    transform_train = []
    transform_test = []

    if data_augmentation=="vanilla":
        print("no data augmentation applied")
    if data_augmentation == "hflip":
        transform_train.append(transforms.RandomHorizontalFlip())
    if data_augmentation == "vflip":
        transform_train.append(transforms.RandomVerticalFlip())
    if data_augmentation == "hvflip":
        transform_train.append(transforms.RandomHorizontalFlip())
        transform_train.append(transforms.RandomVerticalFlip())
    if data_augmentation == "rotate":
        transform_train.append(transforms.RandomRotation(180))

    if dataset_name == "CIFAR10":
        transform_train += [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]
        transform_test += [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_model(model_config):
    return CNN(**model_config)