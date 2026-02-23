from datasets import load_dataset

dataset_full = load_dataset("google-research-datasets/mbpp")

test_data = dataset_full["test"].filter(lambda x: 11 <= x["task_id"] <= 510)

print("Saved test data to mbpp_test with length:", len(test_data))