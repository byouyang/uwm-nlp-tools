"""
Export a small streamed sample from Skylion007/openwebtext to JSONL.

Run:
    python bulk_annotation/get_raw_openweb.py
"""
from datasets import load_dataset
import json

def export_openwebtext_sample():
    # 1. Load the dataset in streaming mode to avoid downloading the whole thing
    print("Loading dataset...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    output_file = "openwebtext_first_50.jsonl"
    
    # 2. Open a JSONL file for writing
    with open(output_file, "w", encoding="utf-8") as f:
        
        # 3. Use .take(50) to grab exactly the first 50 texts
        for i, row in enumerate(dataset.take(50)):
            # Convert the dictionary row to a JSON string and add a newline
            json_line = json.dumps(row, ensure_ascii=False)
            f.write(json_line + "\n")
            
    print(f"Successfully saved 50 texts to {output_file}")

if __name__ == "__main__":
    export_openwebtext_sample()
