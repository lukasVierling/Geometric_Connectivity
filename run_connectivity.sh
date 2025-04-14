#!/bin/bash

# Activate conda environment
source ~/.bashrc
conda activate lora_project

CONFIG_DIR="./configs/vanilla_connectivity_2"
SCRIPT="connectivity.py"

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running: python connectivity.py --config $config"
    python connectivity.py --config "$config"
done

# Wait for any remaining background processes to complete
wait

echo "All runs completed."
