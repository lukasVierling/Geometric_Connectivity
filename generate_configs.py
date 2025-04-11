import os
import yaml

# List of symmetries for which to generate connectivity configs
symmetries = ["hflip", "vflip", "hvflip", "rot90"]

# Interpolation methods to test
methods = ["LMC", "LMCP"]

# Base configuration structure
base_config = {
    "data": {
        # The data_augmentation field will be overwritten with the current symmetry.
        "augment_test": False,
        "batch_size": 256,
        "data_augmentation": "vanilla",
        "data_root": "./data/",
        "dataset": "CIFAR10"
    },
    "interpolation": {
        # The interpolation method will be set per config (LMC or LMCP)
        "num_samples": 10,
        "permutation_iter": 90,
    },
    "models": {
        # Model paths will be updated per symmetry.
        # The vanilla model trained with augmented data is in "<symmetry>_aug"
        # and the symmetry-aware model is in "<symmetry>_symmetry".
        "model_1": None,
        "model_2": None,
        "name": "ResNet18"
    },
    "run": {
        # run_id will be generated below.
        "save_dir": "./logs/connectivity"
    }
}

# Output directory for configs
output_dir = "configs/connectivity_configs_3"
os.makedirs(output_dir, exist_ok=True)

# Generate a config for each symmetry and each interpolation method.
for symmetry in symmetries:
    for method in methods:
        # Create a new config dictionary for this run.
        # We'll re-load the base via a YAML dump/load to ensure a deep copy.
        config = yaml.safe_load(yaml.dump(base_config))
        
        # Set the data augmentation to the current symmetry.
        config["data"]["data_augmentation"] = symmetry
        
        # Set the interpolation method.
        config["interpolation"]["method"] = method
        
        # Update model paths.
        config["models"]["model_1"] = f"./logs/augmentation_experiment/{symmetry}_aug/final_model.pt"
        config["models"]["model_2"] = f"./logs/augmentation_experiment/{symmetry}_symmetry/final_model.pt"
        
        # Construct the run_id.
        config["run"]["run_id"] = f"{symmetry}_aug_to_{symmetry}_symmetry_{method.lower()}"
        
        # Build the output path and write the YAML config file.
        config_filename = f"{config['run']['run_id']}.yaml"
        config_path = os.path.join(output_dir, config_filename)
        with open(config_path, "w") as f:
            yaml.dump(config, f)
            
        print(f"Generated config: {config_path}")

print(f"All configs saved in {output_dir}")
