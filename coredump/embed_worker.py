"""
Embedding Worker — runs in a separate process to avoid Streamlit DLL conflicts.
Called by embedder.py via subprocess.
"""

import sys
import json
import numpy as np

def main():
    """Read codes from stdin or file (JSON), compute embeddings, write to stdout (JSON)."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer

    # Read input — either from a file (for large payloads) or stdin
    input_file = None
    if "--input-file" in sys.argv:
        idx = sys.argv.index("--input-file")
        if idx + 1 < len(sys.argv):
            input_file = sys.argv[idx + 1]

    if input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    else:
        input_data = json.loads(sys.stdin.read())

    codes = input_data['codes']

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Encode
    all_embeddings = []
    batch_size = 32
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, device='cpu')
        all_embeddings.extend(embeddings.tolist())
        # Print progress to stderr
        done = min(i + batch_size, len(codes))
        print(f"PROGRESS:{done}/{len(codes)}", file=sys.stderr, flush=True)

    # Write output
    print(json.dumps(all_embeddings), flush=True)


if __name__ == "__main__":
    main()

