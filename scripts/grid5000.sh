# Load mamba module
module load mamba

# Path to the mamba environment
ENV_PATH="~/.conda/envs/aaf/" # Replace with the actual path to your environment

# # Activate the mamba environment
# source "$ENV_PATH/bin/activate"

# Activate the mamba environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

# The first argument is the script name to run
SCRIPT_NAME="$1"
shift # Shift the arguments to the left, so $2 becomes $1, $3 becomes $2, etc.

bash ./"$SCRIPT_NAME" "$@"