#!/bin/bash

# Parse command-line arguments
while getopts ":c:" opt; do
 case ${opt} in
    c )
      CONFIG_FILE=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      exit 1
      ;;
 esac
done

CONFIG_FILE="configs/fsod/two_branch_10shot_finetuning_dior_pvt_v2_b2_li.yaml"
shift $((OPTIND -1))

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE does not exist."
    exit 1
fi

# Log directory path, constructed based on the configuration file name
LOG_DIR="logs/$(basename "$CONFIG_FILE" .yaml)"

# Check if the log directory exists, create it if not
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Determine the number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Set CUDA_VISIBLE_DEVICES to use all available GPUs
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS-1)))

# Execute the command with the detected number of GPUs and configuration file
python3 fsod_train_net.py --num-gpus $NUM_GPUS --dist-url auto --eval-only --resume \
        --config-file "$CONFIG_FILE" SOLVER.IMS_PER_BATCH 8 2>&1 | tee "$LOG_DIR/$(basename "$CONFIG_FILE" .yaml).log"