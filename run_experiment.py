import os
import sys
import time
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 1. Model Configuration
REPO_ID = "unsloth/Llama-3.2-3B-Instruct-GGUF"
FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
max_tokens = 256
temperature = 0


model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir="models"
)
print(f"Using model path: {model_path}")
# Init Model
# n_gpu_layers=-1 uses Metal on Apple Silicon for all layers
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, 
    n_ctx=2048,
    verbose=False
)

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem step by step. Return the final numerical answer at the end.
If 3 apples cost $1.50, how much do 10 apples cost?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# 3. Define the inference function
def run_inference():
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False
    )
    return output['choices'][0]['text']

# 4. Run Execution
start_time = time.time()
print("Starting generation...")
response = run_inference()
end_time = time.time()

print("\n--- LLM Response ---")
print(response)
print("--------------------\n")

# 5. Summary
print(f"Inference complete in {end_time - start_time:.2f} seconds.")


## sudo ./EnergiBridge/target/release/energibridge -o results.csv --summary .venv/bin/python3 run_experiment.py
