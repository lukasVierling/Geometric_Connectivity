#!/bin/bash

# Activate conda environment
source ~/.bashrc
conda activate lora_project

CONFIG_DIR="configs/vanilla_configs_2"
SCRIPT="main.py"
gpu_index=0
run_count=0

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Run $((run_count + 1)) for $config on GPU $gpu_index"
    CUDA_VISIBLE_DEVICES=$gpu_index python $SCRIPT --config "$config" &

    run_count=$((run_count + 1))

    # Increase GPU index after every 5 runs
    if (( run_count % 1 == 0 )); then
        gpu_index=$((gpu_index + 1))
    fi
done

# Wait for all background processes to finish
wait

echo "All runs completed."
