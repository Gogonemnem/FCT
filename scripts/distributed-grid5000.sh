#!/bin/bash

# Function to get the number of GPUs on a node
get_num_gpus() {
  local node=$1
  oarsh $node nvidia-smi --list-gpus | wc -l
}

# # Function to handle termination signal
# cleanup() {
#   echo "Creating termination signal file on all nodes..."
#   for node in $NODES; do
#     oarsh $node "touch /tmp/terminate_${SCRIPT_NAME}"
#   done
#   echo "Termination signal files created. Waiting for processes to terminate..."
#   wait
#   echo "All processes have been terminated."
# }

# # Trap the interrupt signal (Ctrl+C) and call the cleanup function
# trap cleanup SIGINT

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

# Associative array to store PIDs of background processes
declare -A PIDS

# Execute the specified script on all nodes in parallel
for node in $NODES; do
  echo "Running on node: $node with rank $NODE_RANK"
  MARKER_FILE="/tmp/${SCRIPT_NAME}_${NODE_RANK}_done"
  TERMINATION_FILE="/tmp/terminate_${SCRIPT_NAME}"
  oarsh $node "rm -f $MARKER_FILE $TERMINATION_FILE; cd FCT && module load mamba && mamba activate aaf && bash $SCRIPT_NAME --num-gpus $NUM_GPUS --num-machines $NUM_MACHINES --dist-url $DIST_URL --machine-rank $NODE_RANK $@; if [ -f $TERMINATION_FILE ]; then exit 1; fi; touch $MARKER_FILE" &
  PIDS[$node]=$!  # Store the PID of the background process
  NODE_RANK=$((NODE_RANK + 1))
done

# Wait for all marker files to appear
NODE_RANK=0
for node in $NODES; do
  MARKER_FILE="/tmp/${SCRIPT_NAME}_${NODE_RANK}_done"
  while ! oarsh $node "test -f $MARKER_FILE"; do
    sleep 5
  done
  NODE_RANK=$((NODE_RANK + 1))
done

echo "All distributed training processes have completed."
