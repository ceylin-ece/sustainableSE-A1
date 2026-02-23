#!/bin/bash

# Navigate to the project directory relative to the script
cd "$(dirname "$0")" || { echo "Failed to navigate to project directory"; exit 1; }
PROJECT_DIR="$(pwd)"

MODELS=(
    "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    "models/Phi-3-mini-4k-instruct-q4.gguf"
    "models/qwen2.5-3b-instruct-q4_k_m.gguf"
)

START_TRIAL=${1:-1}
END_TRIAL=${2:-20}
SUMMARY_CSV="results/total_energy_summary.csv"

# Ensure output folders exist
mkdir -p "results"

# Initialize the summary CSV with headers if it doesn't exist
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "Model_Name,Trial_Number,Total_Joules,Total_Execution_Time_Sec" > "$SUMMARY_CSV"
fi

echo "Starting trials from $START_TRIAL to $END_TRIAL for all models..."
echo "Results will be saved in $PROJECT_DIR/results"
echo "Summary energy will be aggregated into $SUMMARY_CSV"
echo "======================================================"

for trial in $(seq $START_TRIAL $END_TRIAL); do
    echo ""
    echo "======================================================"
    echo "                 STARTING TRIAL $trial                  "
    echo "======================================================"
    
    for model in "${MODELS[@]}"; do
        
        # Extract the base model name
        base_name=$(basename "$model")
        clean_name="${base_name%.gguf}"
        
        echo "--> Running $clean_name (Trial $trial)..."
        
        # We run the trial script and capture its STDOUT to a temporary text file
        # so we can parse the exact energy footprint printed by EnergiBridge
        tmp_output="tmp_run_output.txt"
        sh run_trial.sh "$model" "$trial" | tee "$tmp_output"
        
        # Search the temporary output for the specific EnergiBridge summary line
        # Expected format: "Energy consumption in joules: 90.29885560989379 for 3.666201 sec of execution."
        summary_line=$(grep "Energy consumption in joules:" "$tmp_output")
        
        if [ ! -z "$summary_line" ]; then
            # Extract the actual numbers using awk
            joules=$(echo "$summary_line" | awk '{print $5}')
            time_sec=$(echo "$summary_line" | awk '{print $7}')
            
            # Append to our master summary CSV
            echo "$clean_name,$trial,$joules,$time_sec" >> "$SUMMARY_CSV"
            echo "    -> Logged Energy: $joules J over $time_sec s."
        else
            echo "    -> Error: Could not find energy summary in output for $clean_name!"
        fi
        
        # Clean up temp file
        rm -f "$tmp_output"
        
    done
done

echo "======================================================"
echo "Trials $START_TRIAL to $END_TRIAL finished. Summary saved to $SUMMARY_CSV."
