#!/bin/bash
cd ..

# Function to get the number of GPUs on a node
get_num_gpus() {
  local node=$1
  oarsh $node nvidia-smi --list-gpus | wc -l
}

# Parse the script name
SCRIPT_NAME=$1
shift

# Get available nodes
NODES=$(oarprint host | uniq)

# Determine the number of nodes
NUM_MACHINES=$(echo "$NODES" | wc -l)

# Determine the first node for dist-url
FIRST_NODE=$(echo "$NODES" | head -n 1)
DIST_URL="tcp://$FIRST_NODE:29500"

# Get the number of GPUs on the first node
NUM_GPUS=$(get_num_gpus $FIRST_NODE)

echo "Distributed training on $NUM_MACHINES machines with $NUM_GPUS GPUs each."
echo "Using dist-url: $DIST_URL"

# Initialize node rank counter
NODE_RANK=0

# Execute the specified script on all nodes in parallel
echo "$NODES" | while read node; do
  echo "Running on node: $node with rank $NODE_RANK"
  oarsh $node "cd FCT && module load mamba && mamba activate aaf && bash $SCRIPT_NAME --num-gpus $NUM_GPUS --num-machines $NUM_MACHINES --dist-url $DIST_URL --machine-rank $NODE_RANK $@" &
  NODE_RANK=$((NODE_RANK + 1))
done

# Wait for all background processes to finish
wait

echo "All distributed training processes have completed."
