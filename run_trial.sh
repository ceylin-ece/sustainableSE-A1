#!/bin/bash

# Navigate to the project directory relative to the script
cd "$(dirname "$0")" || { echo "Failed to navigate to project directory"; exit 1; }

# Require 2 arguments: the model path, and the trial number
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_trial.sh <model_path> <trial_number>"
    exit 1
fi

MODEL_PATH=$1
TRIAL_NUM=$2

echo "Running trial $TRIAL_NUM with model: $MODEL_PATH"
echo "========================================"

# Extract the filename without the path, then without the .gguf to create a clean name
base_name=$(basename "$MODEL_PATH")
clean_name="${base_name%.gguf}"

# Define output folders
EVAL_DIR="results/eval"
ENERGY_DIR="results/energy"

# Ensure output folders exist
mkdir -p "$EVAL_DIR"
mkdir -p "$ENERGY_DIR"

# Define output files for this specific model and trial in their respective folders
eval_result_file="${EVAL_DIR}/eval_results_${clean_name}_trial_${TRIAL_NUM}.csv"
energy_result_file="${ENERGY_DIR}/energy_results_${clean_name}_trial_${TRIAL_NUM}.csv"

# Run the python script with the specific virtual environment python
# We pass the trial number and the new eval results csv output flag
sudo ./EnergiBridge/target/release/energibridge -o "$energy_result_file" --summary \
    .venv/bin/python3 run_experiment.py --model_path "$MODEL_PATH" --limit 500 --trial "$TRIAL_NUM" --output_csv "$eval_result_file"

if [ $? -ne 0 ]; then
    echo "Trial $TRIAL_NUM failed for $MODEL_PATH!"
fi
echo ""
