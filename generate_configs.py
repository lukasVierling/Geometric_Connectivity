#!/usr/bin/env python3
import os
import yaml

def generate_configs():
    # Base structure for each config
    base_config = {
        "data": {
            "augment_test": False,
            "batch_size": 256,
            "data_augmentation": "vanilla",
            "data_root": "./data/",
            "dataset": "CIFAR10"
        },
        "interpolation": {
            # "method" will be set below
            "num_samples": 10,
            "permutation_iter": 90
        },
        "models": {
            # "model_1" and "model_2" to be set; name is fixed.
            "name": "ResNet18"
        },
        "run": {
            # "run_id" and "save_dir" to be set below
        }
    }
    
    # Methods and symmetries as specified
    methods = ["LMC", "LMCP"]
    symmetries = ["vanilla","hflip", "vflip", "hvflip", "rot90"]

    # List to collect all configs (each as tuple: (filename, config_dict))
    configs = []
    
    # We iterate over each method and then iterate over unordered symmetry pairs.
    # By setting the inner loop to start at the current outer index,
    # we ensure that if A_to_B is produced, then B_to_A is skipped.
    for method in methods:
        for i, sym1 in enumerate(symmetries):
            for sym2 in symmetries[i:]:
                # Start with the base copy; use dict.copy() for a shallow copy
                # Note: nested dictionaries need to be re-copied if modified.
                cfg = {
                    "data": base_config["data"].copy(),
                    "interpolation": base_config["interpolation"].copy(),
                    "models": base_config["models"].copy(),
                    "run": {}
                }
                # Set the interpolation method
                cfg["interpolation"]["method"] = method

                # Determine model paths:
                # For self loops (same symmetry) use model indices 1 and 2.
                # Otherwise, use index 1 for both.
                if sym1 == sym2:
                    model1 = f"./logs/vanilla/{sym1}_1/final_model.pt"
                    model2 = f"./logs/vanilla/{sym2}_2/final_model.pt"
                    run_id = f"{sym1}_1_to_{sym2}_2_{method.lower()}"
                else:
                    model1 = f"./logs/vanilla/{sym1}_1/final_model.pt"
                    model2 = f"./logs/vanilla/{sym2}_1/final_model.pt"
                    run_id = f"{sym1}_1_to_{sym2}_1_{method.lower()}"

                cfg["models"]["model_1"] = model1
                cfg["models"]["model_2"] = model2

                # Set the run fields: run_id and save_dir (save_dir depends on the method)
                cfg["run"]["run_id"] = run_id
                if method == "LMC":
                    cfg["run"]["save_dir"] = "./logs/vanilla_connectivity"
                else:
                    cfg["run"]["save_dir"] = "./logs/connectivity"

                # Create a filename for the config file
                filename = f"configs/vanilla_connectivity/{run_id}.yaml"
                configs.append((filename, cfg))
    
    # Write each config dictionary to its corresponding YAML file
    for filename, conf in configs:
        with open(filename, "w") as f:
            # Dump the config preserving the structure
            yaml.dump(conf, f, sort_keys=False)
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_configs()
