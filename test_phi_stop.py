import sys
from llama_cpp import Llama

llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1, 
    n_ctx=2048,
    verbose=False
)

system_prompt = "You are an expert Python programmer. Write a Python function solving the requested task. Return ONLY the raw python code block."
task_text = "Write a python function to find the volume of a triangular prism."

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text}
    ],
    max_tokens=512,
    temperature=0.0
)

print("--- GENERATED OUTPUT ---")
print(output['choices'][0]['message']['content'])
print("--- REPR OUTPUT ---")
print(repr(output['choices'][0]['message']['content']))
