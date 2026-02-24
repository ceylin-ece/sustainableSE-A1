import os
import sys
import time
import json
import argparse
import re
import csv
import signal
from llama_cpp import Llama

# 1. Configuration
parser = argparse.ArgumentParser(description="Run LLM experiment on MBPP.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the GGUF model")
parser.add_argument("--dataset", type=str, default="datasets/mbpp_test_11_to_510.jsonl", help="Path to JSONL dataset")
parser.add_argument("--limit", type=int, default=500, help="Number of tasks to evaluate (for testing)")
parser.add_argument("--trial", type=int, default=1, help="Trial number")
parser.add_argument("--output_csv", type=str, default="", help="Path to output CSV for tracking task accuracy")
args = parser.parse_args()

model_path = args.model_path

print(f"Using model path: {model_path}")
if not os.path.exists(model_path):
    print(f"Error: Model path {model_path} does not exist!")
    sys.exit(1)

# Initialize CSV if output path is provided
if args.output_csv:
    # Write headers if file doesn't exist
    if not os.path.exists(args.output_csv):
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model_Name', 'Trial', 'Task_ID', 'Passed', 'Reason', 'Duration_Sec', 'Prompt_Tokens', 'Completion_Tokens', 'Total_Tokens', 'Prompt'])

# 2. Init Model
# n_gpu_layers=-1 uses Metal on Apple Silicon for all layers
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, 
    n_ctx=2048,
    verbose=False
)

# Helper to cleanly extract python code if the LLM adds markdown or text
def extract_code(text):
    match = re.search(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    code = match.group(1).strip() if match else text.strip()
    
    # Strip example usage to avoid side effects
    for marker in ["if __name__ == '__main__':", "# Example usage", "# Test cases", "print("]:
        if marker in code:
            code = code.split(marker)[0].strip()
            
    return code

# Safely evaluate LLM code output
def evaluate_code(generated_code, test_list, test_setup, test_block, test_imports):
    # Combine everything: setup, generated code, and the asserts
    imports_code = "\n".join(test_imports) if test_imports else ""
    full_code = f"{imports_code}\n{test_setup}\n{generated_code}\n"
    
    if test_block:
        full_code += test_block + "\n"
    else:
        for test in test_list:
            full_code += test + "\n"
        
    namespace = {}
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out (possible infinite loop in generated code)")
        
    # Set a 5-second timeout to prevent infinite loops from hanging the evaluation
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    
    try:
        # Note: using exec for user-generated code is risky in production, but fine for local benchmarking
        exec(full_code, namespace)
        return True, "Passed"
    except TimeoutError as e:
        return False, f"TimeoutError: {str(e)}"
    except AssertionError:
        return False, "AssertionError"
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return False, f"Exception: {type(e).__name__} - {str(e)}\n{error_msg}"
    finally:
        signal.alarm(0) # Disable the alarm

# 3. Execution Loop
correct = 0
total = 0
start_time = time.time()
base_model_name = os.path.basename(model_path)

with open(args.dataset, 'r') as f:
    for line in f:
        if total >= args.limit:
            break
            
        task_start = time.time()
        
        task = json.loads(line)
        task_id = task.get("task_id")
        task_text = task.get("prompt", task.get("text", "")) # handle evalplus or original mbpp
        test_list = task.get("test_list", [])
        test_setup_code = task.get("test_setup_code", "")
        test_block = task.get("test", "")
        test_imports = task.get("test_imports", [])
        
        # Extract the expected function name and signature
        signature = "def unknown_function():"
        expected_func = "unknown"
        func_match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):", task.get("code", ""))
        if func_match:
            expected_func = func_match.group(1)
            signature = func_match.group(0)
            
        system_prompt = (
            "You are an expert Python programmer. "
            "Write a Python function solving the requested task. "
            f"CRITICAL: Start your code exactly with this signature: `{signature}`\n"
            "DO NOT add any examples, test cases, or usage. Return ONLY the raw python code block."
        )
        
        # We use create_chat_completion to automatically apply the proper system/user prompt formatting
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_text}
            ],
            max_tokens=512,
            temperature=0.0
        )
        
        response_text = output['choices'][0]['message']['content']
        code = extract_code(response_text)
        
        prompt_tokens = output.get('usage', {}).get('prompt_tokens', 0)
        completion_tokens = output.get('usage', {}).get('completion_tokens', 0)
        total_tokens = output.get('usage', {}).get('total_tokens', 0)
        
        # Programmatically enforce the correct function name in case the AI ignored instructions
        if expected_func != "unknown":
            code = re.sub(r"def\s+[a-zA-Z0-9_]+\s*\(", f"def {expected_func}(", code, count=1)
        
        passed, reason = evaluate_code(code, test_list, test_setup_code, test_block, test_imports)
        
        task_end = time.time()
        task_duration = task_end - task_start
        
        if passed:
            correct += 1
            print(f"Task {task_id}: PASS")
        else:
            print(f"Task {task_id}: FAIL ({reason.splitlines()[0]})")
            
        # Log to CSV
        if args.output_csv:
            with open(args.output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Model, Trial, Task_ID, Passed, Reason, Duration, Prompt_Tokens, Completion_Tokens, Total_Tokens, Prompt
                writer.writerow([base_model_name, args.trial, task_id, passed, reason.splitlines()[0], f"{task_duration:.2f}", prompt_tokens, completion_tokens, total_tokens, task_text])
                
        total += 1

end_time = time.time()

# 4. Summary
print("\n=== Experiment Summary ===")
print(f"Model: {base_model_name}")
print(f"Trial: {args.trial}")
print(f"Total evaluated: {total}")
print(f"Total correct: {correct}")
print(f"Accuracy: {(correct / total) * 100:.2f}%" if total > 0 else "Accuracy: N/A")
print(f"Time taken: {end_time - start_time:.2f} seconds.")
print("==========================\n")
