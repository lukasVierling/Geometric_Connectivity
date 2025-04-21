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
import random
import numpy as np

#my imports
from models.ResNet import ResNet18

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model_and_config(model, config, save_dir, name=None):

    os.makedirs(save_dir, exist_ok=True)

    if not name:
        name = "final_model"
    
    model_path = os.path.join(save_dir, f"{name}.pt")
    torch.save(model.state_dict(), model_path)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class Random90Rotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, angle)

def get_data_loaders(dataset_config):
    dataset_name = dataset_config["dataset"]
    batch_size = dataset_config["batch_size"]
    data_root = dataset_config.get("data_root", "./data")
    data_augmentation = dataset_config["data_augmentation"]
    augment_test = dataset_config["augment_test"]
    if augment_test:
        print("Augmenting test data.")
    transform_train = []
    transform_test = []

    if data_augmentation=="vanilla":
        print("no data augmentation applied")
    elif data_augmentation == "hflip":
        transform_train.append(transforms.RandomHorizontalFlip())
        if augment_test:
            transform_test.append(transforms.RandomHorizontalFlip())
    elif data_augmentation == "vflip":
        transform_train.append(transforms.RandomVerticalFlip())
        if augment_test:
            transform_test.append(transforms.RandomVerticalFlip())
    elif data_augmentation == "hvflip":
        transform_train.append(transforms.RandomHorizontalFlip())
        transform_train.append(transforms.RandomVerticalFlip())
        if augment_test:
            transform_test.append(transforms.RandomHorizontalFlip())
            transform_test.append(transforms.RandomVerticalFlip())
    elif data_augmentation == "rotate":
        transform_train.append(Random90Rotation())
        if augment_test:
            transform_test.append(Random90Rotation())
    elif data_augmentation == "ResNet":
        print("Using ResNet augmentation")
        '''
        use the data augmentation emplyoed by ResNet paper and We use a standard data augmentation scheme (Lin et al., 2013; Romero et al., 2014; Lee
        et al., 2015; Springenberg et al., 2014; Srivastava et al., 2015; Huang et al., 2016b; Larsson et al.,
        2016), in which the images are zero-padded with 4 pixels on each side, randomly cropped to produce
        32×32 images, and horizontally mirrored with probability 0.5. -> Snapshot ensembles paper
        '''
        transform_train.append(transforms.RandomCrop(32, padding=4))
        transform_train.append(transforms.RandomHorizontalFlip())
        if augment_test:
            transform_test.append(transforms.RandomCrop(32, padding=4))
            transform_test.append(transforms.RandomHorizontalFlip())
    else:
        print("Unknown data augmentation:", data_augmentation)

    if dataset_name == "CIFAR10":
        # norm values from https://www.mindspore.cn/tutorials/application/en/r2.1/cv/resnet50.html
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
    name = model_config["name"]
    del model_config["name"]
    if name == "ResNet18":
        return ResNet18(**model_config)
    else:
        print(f"Model {name} not supported. Return ResNet18.")
        return ResNet18(**model_config)