import os
from huggingface_hub import snapshot_download

# List of Hugging Face repositories requested
MODELS = [
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "microsoft/Phi-3-mini-4k-instruct-gguf"
]

# Destination directory relative to where the script is run
DEST_DIR = "models"

def download_models():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    for repo_id in MODELS:
        print(f"\n[{repo_id}]")
        print("Finding and downloading Q4_K_M GGUF model...")
        
        try:
            # We use snapshot_download to download matching filenames (Q4_K_M quantization)
            # local_dir_use_symlinks=False ensures the actual file is copied to the models dir,
            # avoiding the broken symlink issue you had earlier!
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=DEST_DIR,
                allow_patterns=["*Q4_K_M.gguf", "*q4_k_m.gguf", "*q4.gguf"],
                local_dir_use_symlinks=False 
            )
            print(f"Successfully downloaded to: {downloaded_path}")
        except Exception as e:
            print(f"Error downloading from {repo_id}: {e}")

if __name__ == "__main__":
    print(f"Downloading models to '{DEST_DIR}' directory...")
    download_models()
    print("\nAll downloads completed.")
