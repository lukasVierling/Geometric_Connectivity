#!/bin/bash

CONFIG_DIR="./configs/connectivity_configs_3"

for config in "$CONFIG_DIR"/*.yaml; do
  echo "Running: python connectivity.py --config $config"
  python connectivity.py --config "$config"
done
