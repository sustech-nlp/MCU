# %%
"""
Add model accuracy scores to wmdp_deduped_bio and wmdp_deduped_cyber datasets
"""
import json
from pathlib import Path

import torch as pt
from datasets import Dataset
from transformers import AutoModelForCausalLM

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root

# ===== Configuration Parameters =====
# MODEL_PATH = "allenai/Olmo-3-1025-7B"  # Model path
# MODEL_KEY = "Olmo-3-1025-7B"  # Field name to store in data
# MODEL_PATH = "google/gemma-2-9b"  # Model path
# MODEL_KEY = "gemma-2-9b"  # Field name to store in data
# DEVICE = "cuda:3"  # Device to use
MODEL_PATH = "HuggingFaceH4/zephyr-7b-beta"  # Model path
MODEL_KEY = "zephyr-7b-beta"  # Field name to store in data
DEVICE = "cuda:0"  # Device to use
# =====================================

pt.set_default_device(DEVICE)

# %%
print(f"Loading model from: {MODEL_PATH}")
print(f"Will save results under key: {MODEL_KEY}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=pt.bfloat16, device_map=DEVICE
)

# %%
general_data_path = repo_root() / "data"

# List of files to process
files_to_process = []
for category in ["wmdp_deduped_bio", "wmdp_deduped_cyber"]:
    category_path = general_data_path / category
    for jsonl_file in category_path.glob("*.jsonl"):
        files_to_process.append(jsonl_file)

print(f"Found {len(files_to_process)} files to process:")
for f in files_to_process:
    print(f"  {f}")

# %%
def process_file(file_path: Path):
    """Add accuracy scores for the specified model to a single file"""
    print(f"\nProcessing: {file_path}")
    
    # Load all data
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    # Check if the model field already exists
    if data and MODEL_KEY in data[0]:
        print(f"  Skipping: already has {MODEL_KEY} field")
        return
    
    # Check if Llama-3.1-8B field exists (indicates this is a question file with accuracy)
    if not data or "Llama-3.1-8B" not in data[0]:
        print(f"  Skipping: no Llama-3.1-8B field (not a question file with accuracy)")
        return
    
    # Calculate accuracy for each question
    updated_data = []
    for i, ex in enumerate(data):
        print(f"  [{i+1}/{len(data)}] {ex['question'][:50]}...")
        
        # Create temporary Dataset for evaluation
        ds = Dataset.from_list([ex])
        acc = eval_on(ds, model, temperature=1)
        
        # Add accuracy
        ex[MODEL_KEY] = acc
        updated_data.append(ex)
        print(f"    Accuracy: {acc}")
    
    # Write back to file
    with open(file_path, "w") as f:
        for ex in updated_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"  Done: updated {len(updated_data)} examples")

# %%
# Process all files
for file_path in files_to_process:
    process_file(file_path)

print("\n\nAll files processed!")
