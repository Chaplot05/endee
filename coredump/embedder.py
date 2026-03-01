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

    For large payloads (>1MB), data is passed via a temp file to avoid
    Windows pipe buffer deadlocks that cause [Errno 32] Broken pipe.

    Args:
        codes: List of source code strings
        progress_callback: Optional callable(current, total)

    Returns:
        list[list[float]]: 384-dim embedding vectors
    """
    import threading
    import tempfile

    input_data = json.dumps({"codes": codes})
    use_tempfile = len(input_data) > 1_000_000  # >1MB → use temp file

    tmp_path = None
    try:
        if use_tempfile:
            # Write input to a temp file to sidestep pipe buffer limits
            fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="coredump_embed_")
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(input_data)

            proc = subprocess.Popen(
                [sys.executable, _WORKER_PATH, "--input-file", tmp_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, _WORKER_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            # Write stdin in a background thread to avoid deadlock.
            # On Windows, pipe buffers are small (~4KB). If the parent writes
            # a large payload synchronously while the child hasn't started
            # reading, the parent blocks on write — classic pipe deadlock
            # that surfaces as [Errno 32] Broken pipe.
            stdin_error = [None]

            def write_stdin():
                try:
                    proc.stdin.write(input_data)
                    proc.stdin.close()
                except Exception as e:
                    stdin_error[0] = e
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

            stdin_thread = threading.Thread(target=write_stdin, daemon=True)
            stdin_thread.start()

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
        if not use_tempfile:
            stdin_thread.join(timeout=30)
        stderr_thread.join(timeout=5)

        # Check for stdin write failure (pipe-based mode only)
        if not use_tempfile and stdin_error[0] is not None:
            error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else str(stdin_error[0])
            raise RuntimeError(
                f"Embedding worker crashed before reading input:\n{error_msg}"
            )

        if proc.returncode != 0:
            error_msg = "\n".join(stderr_lines[-10:])
            raise RuntimeError(f"Embedding worker failed (exit code {proc.returncode}):\n{error_msg}")

        try:
            embeddings = json.loads(stdout_data)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse embeddings output: {e}\nStderr: {chr(10).join(stderr_lines[-5:])}")

        return embeddings

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

