import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import argparse

# my imports
from mode_connect import probe_connect, train_quad_midpoint
from utils.utils import get_data_loaders, load_config
from models.ResNet import ResNet18, create_model
from find_permutation import res_permute


def test_connectivity(config_path):
    config = load_config(config_path)
    model_config = config["models"]
    interpolation_config = config["interpolation"]
    data_config = config["data"]
    run_config = config["run"]
    print("Start Connectivity Testing with the following Settings:")
    print("----------------------------------------")
    print("Model Config: \n")
    print(model_config)
    print("----------------------------------------")
    print("Interpolation Config: \n")
    print(interpolation_config)
    print("----------------------------------------")
    print("Data Config: \n")
    print(data_config)
    print("----------------------------------------")
    print("Run Config: \n")
    print(run_config)
    print("----------------------------------------")

    
    #get the run args
    save_dir = run_config["save_dir"]
    run_id = run_config["run_id"]

    #get the model args
    model_1_path = model_config["model_1"]
    model_2_path = model_config["model_2"]
    model_name = model_config["name"]

    #set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #prepare the models

    #load the models
    model_1 = ResNet18()
    model_2 = ResNet18()

    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    # turn convolutions into normal convolutions
    model_1 = model_1.to_normal()
    model_2 = model_2.to_normal()

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    model_middle = None

        #get the interpolation args
    interpolation_method = interpolation_config["method"]
    num_samples = interpolation_config["num_samples"]
    permutation_iter = interpolation_config["permutation_iter"]
    if interpolation_method == "QMC":
        print("Using QMC...")
        train_setup = {
            "model": "ResNet18",
            "num_classes": 10,
            "n_epochs": 20,
            "start_lr": 0.1,
            "lr_decay": 0.1,
            "decay_epochs": 40,
            "data_config": data_config
        }
        print("Train the middle model...")
        #get the middle model for quadratic interpolation
        model_middle = train_quad_midpoint(model_1, model_2, train_setup)
    if interpolation_method == "LMCP":
        print("Using LMCP...")
        #get data loader and then find the optimal permutation of model_1
        train_loader,_ = get_data_loaders(data_config)
        model_1 = res_permute(net_permuted=model_1, net_target=model_2, dataloader=train_loader, n_match_iters=permutation_iter)
    if interpolation_method == "LMC":
        print("Using LMC...")
        #do nothing

    setup = {
    "model_class": model_name,
    "connect_pattern": interpolation_method,
    "num_classes": 10,
    "n_interpolations": num_samples,
    "seed_id": 0,
    "data_config": data_config
    }

    # use smaller subset ... TODO later
    accs, losses = probe_connect(model_1=model_1.eval(), model_2=model_2.eval(),
                                                    net_midpoint=None if model_middle is None else model_middle.eval(),
                                                    setup=setup, return_loss=True)
    
    results = {"accs": accs, "losses": losses}
    
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, run_id)


    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="connectivity.yaml")
    args = parser.parse_args()
    config_path = args.config
    test_connectivity(config_path)
    
