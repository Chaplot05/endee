"""
Code Embedding Engine
Delegates to embed_worker.py via subprocess to avoid
torch DLL initialization issues inside Streamlit on Windows.

Produces 384-dimensional normalized embeddings for all-MiniLM-L6-v2.
"""

import os
import sys
import json
import subprocess

EMBEDDING_DIM = 384
_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_worker.py")

def get_model():
    """No-op. We delegate to the worker process."""
    return True

def embed_chunk(code: str) -> list:
    """Generate embedding for a single code chunk."""
    result = embed_batch([code])
    return result[0]

def embed_batch(codes: list, progress_callback=None) -> list:
    """Generate embeddings by calling embed_worker.py in a subprocess.

    This avoids loading torch inside Streamlit's script runner,
    which crashes on Windows due to DLL initialization issues.

    Args:
        codes: List of source code strings
        progress_callback: Optional callable(current, total)

    Returns:
        list[list[float]]: 384-dim embedding vectors
    """
    input_data = json.dumps({"codes": codes})

    proc = subprocess.Popen(
        [sys.executable, _WORKER_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    proc.stdin.write(input_data)
    proc.stdin.close()

    import threading
    stderr_lines = []

    def read_stderr():
        for line in proc.stderr:
            line = line.strip()
            stderr_lines.append(line)
            if line.startswith("PROGRESS:") and progress_callback:
                try:
                    parts = line.replace("PROGRESS:", "").split("/")
                    progress_callback(int(parts[0]), int(parts[1]))
                except Exception:
                    pass

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    stdout_data = proc.stdout.read()
    proc.wait()
    stderr_thread.join(timeout=5)

    if proc.returncode != 0:
        error_msg = "\n".join(stderr_lines[-10:])
        raise RuntimeError(f"Embedding worker failed (exit code {proc.returncode}):\n{error_msg}")

    try:
        embeddings = json.loads(stdout_data)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse embeddings output: {e}\nStderr: {chr(10).join(stderr_lines[-5:])}")

    return embeddings
