import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import yaml
from tqdm import tqdm
import wandb
#my imports
from utils.eval import evaluate
from utils.train import train_epoch
from utils.utils import load_config, get_data_loaders, get_model, save_model_and_config, set_seed

from mmc_utils import LR_Scheduler

import os

def main(config_path):
    config = load_config(config_path)

    #get sub configs
    model_config = config["model"]
    train_config = config["train"]
    data_config = config["data"]
    run_config = config["run"]

    print("Start Training with the following Settings:")
    print("----------------------------------------")
    print("Model Config: \n")
    print(model_config)
    print("----------------------------------------")
    print("Train Config: \n")
    print(train_config)
    print("----------------------------------------")
    print("Data Config: \n")
    print(data_config)
    print("----------------------------------------")
    print("Run Config: \n")
    print(run_config)
    print("----------------------------------------")

    #get run settings
    wandb_name = run_config["wandb_name"]
    run_id = run_config["id"]
    save_path = run_config["save_path"]
    seed = run_config.get("seed", 0)
    save_init = run_config.get("save_init", False)
    track_symmetry = run_config.get("track_symmetry", False)

    #setting seed
    set_seed(seed)
    print(f"Set seed for the run: {seed}.")

    print(f"Initialize Weights and Biases with name: {wandb_name}")

    #get train args
    epochs = train_config["epochs"]
    epochs = train_config["epochs"]
    momentum = train_config["momentum"]
    weight_decay = train_config["weight_decay"]
    base_lr = train_config["base_lr"]
    final_lr = train_config["final_lr"]

    #setup wandb
    wandb.init(project=wandb_name, name=run_id, config=config)

    #create save dir
    save_dir = os.path.join(save_path, run_id)

    #dataset
    train_loader ,test_loader = get_data_loaders(data_config)

    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train on device: {device}")

    #model
    model = get_model(model_config)
    model = model.to(device)

    #load pretrained model
    if run_config.get("from_pretrained", False):
        print("Init from pretrained")
        init_model = torch.load(run_config["pretrained_path"])
        model.load_state_dict(init_model)
    
    #save model if set
    if save_init:
        save_model_and_config(model, config, save_dir, name="init")

    #optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = F.cross_entropy
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    #scheduler = LR_Scheduler(optimizer, epochs, base_lr, final_lr, len(train_loader))
    all_symmetries = []
    print("Start Training...")
    for epoch in tqdm(range(epochs), desc=f"Training for {epochs} epochs..."):
        if epoch % 1 == 0 and track_symmetry: #1 can be adjusted to track symmetry less often
            symmetries = model.evaluate_symmetry()
            wandb.log({ 
                    "horizontal": symmetries[0],
                    "vertical":symmetries[1],
                    "horizontal-vertical": symmetries[2],
                    "rot90": symmetries[3],
                    "total_symmetry": symmetries[4],
                    "mean_total_symmetry": symmetries[5],
            })
            all_symmetries.append(symmetries)
        train_epoch(model, optimizer, None, criterion, train_loader, device, epoch) #scheduler None because we have epoch-wise scheduler
        evaluate(model, test_loader, device, epoch)
        scheduler.step()
        
    
    print("Finished Training!")

    if track_symmetry:
        config["symmetries"] = all_symmetries

    save_model_and_config(model, config, save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config_path = args.config
    main(config_path)