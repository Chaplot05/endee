"""
Code Embedding Engine
Produces 384-dimensional embeddings for code chunks.

Strategy (in order):
  1. In-process sklearn TF-IDF + SVD (fast, no torch, no subprocess, no API)
     — works everywhere including Render free tier
  2. Falls back to local sentence-transformers subprocess if available

The TF-IDF approach captures "meaning" via term frequency patterns in code.
While not as rich as transformer embeddings, it is fully deterministic,
instant, and sufficient for drift / similarity analysis.
"""

import os
import sys
import json
import numpy as np

EMBEDDING_DIM = 384

_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_worker.py")


def get_model():
    """No-op kept for API compatibility with app.py."""
    return True


def embed_chunk(code: str) -> list:
    """Generate embedding for a single code chunk."""
    result = embed_batch([code])
    return result[0]


# ── Lightweight in-process embeddings (sklearn) ─────────────
def _sklearn_embed(codes: list, progress_callback=None) -> list:
    """
    Generate embeddings using TF-IDF + Truncated SVD (LSA).
    
    This produces dense semantic vectors from code text without
    needing torch or any external API.  Fully deterministic.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    import re

    # Preprocess: tokenise code into meaningful tokens
    def tokenize_code(code: str) -> str:
        # Split camelCase and snake_case
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = code.replace('_', ' ')
        # Keep alphanumeric and spaces
        code = re.sub(r'[^a-zA-Z0-9\s]', ' ', code)
        # Collapse whitespace
        code = re.sub(r'\s+', ' ', code).strip().lower()
        return code

    processed = [tokenize_code(c) for c in codes]

    # TF-IDF with code-friendly settings
    n_components = min(EMBEDDING_DIM, len(codes) - 1) if len(codes) > 1 else 1
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # log-scale TF
        min_df=1,
        max_df=0.95,
        token_pattern=r'(?u)\b\w+\b',
    )

    tfidf_matrix = vectorizer.fit_transform(processed)
    
    if progress_callback:
        progress_callback(len(codes) // 3, len(codes))

    # Reduce to EMBEDDING_DIM with SVD (Latent Semantic Analysis)
    if n_components < EMBEDDING_DIM:
        # If we have fewer docs than EMBEDDING_DIM, pad with zeros
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)
        # Pad to full EMBEDDING_DIM
        padded = np.zeros((reduced.shape[0], EMBEDDING_DIM), dtype=np.float32)
        padded[:, :reduced.shape[1]] = reduced
        reduced = padded
    else:
        svd = TruncatedSVD(n_components=EMBEDDING_DIM, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)

    if progress_callback:
        progress_callback(2 * len(codes) // 3, len(codes))

    # L2 normalize
    reduced = normalize(reduced, norm='l2')

    if progress_callback:
        progress_callback(len(codes), len(codes))

    return reduced.tolist()


# ── Local subprocess fallback ───────────────────────────────
def _local_embed(codes: list, progress_callback=None) -> list:
    """Fallback: run embed_worker.py as a subprocess (needs sentence-transformers)."""
    import subprocess
    import threading
    import tempfile

    input_data = json.dumps({"codes": codes})
    use_tempfile = len(input_data) > 1_000_000

    tmp_path = None
    try:
        if use_tempfile:
            fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="coredump_embed_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(input_data)
            proc = subprocess.Popen(
                [sys.executable, _WORKER_PATH, "--input-file", tmp_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, _WORKER_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
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

        if not use_tempfile and stdin_error[0] is not None:
            error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else str(stdin_error[0])
            raise RuntimeError(f"Embedding worker crashed:\n{error_msg}")

        if proc.returncode != 0:
            error_msg = "\n".join(stderr_lines[-10:])
            raise RuntimeError(f"Embedding worker failed (exit {proc.returncode}):\n{error_msg}")

        try:
            embeddings = json.loads(stdout_data)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bad worker output: {e}\nStderr: {chr(10).join(stderr_lines[-5:])}")

        return embeddings

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── Main batch function ─────────────────────────────────────
def embed_batch(codes: list, progress_callback=None) -> list:
    """Generate embeddings for a list of code strings.

    Uses in-process TF-IDF + SVD (no torch needed).
    Falls back to sentence-transformers subprocess if available
    and USE_LOCAL_TRANSFORMER env var is set.

    Args:
        codes: List of source code strings
        progress_callback: Optional callable(current, total)

    Returns:
        list[list[float]]: 384-dim embedding vectors
    """
    if not codes:
        return []

    # Check if user explicitly wants the heavy transformer model
    use_transformer = os.environ.get("USE_LOCAL_TRANSFORMER", "").lower() in ("1", "true", "yes")

    if use_transformer:
        try:
            return _local_embed(codes, progress_callback)
        except Exception as e:
            print(f"[embedder] Local transformer failed ({e}), falling back to TF-IDF", flush=True)

    # Default: lightweight TF-IDF + SVD (works everywhere)
    return _sklearn_embed(codes, progress_callback)
