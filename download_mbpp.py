from datasets import load_dataset

dataset_full = load_dataset("evalplus/mbppplus")

test_data = dataset_full["test"].filter(lambda x: 11 <= x["task_id"] <= 510)

import os

# Create datasets directory if it doesn't exist
output_dir = "/Users/kunalnarwani/Desktop/Delft/Delft/Sem_1/Sustainable Software Engineering/P1/GreenLLM_Experiment/sustainableSE-A1/datasets"
os.makedirs(output_dir, exist_ok=True)

# Save as a json lines file since it's easy to read line-by-line later
output_file = os.path.join(output_dir, "mbpp_test_11_to_510.jsonl")
test_data.to_json(output_file, orient="records", lines=True)

print(f"Saved {len(test_data)} MBPP test records to {output_file}")