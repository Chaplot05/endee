"""
Git History Walker — Multi-Language
Clones a public GitHub repo, walks the last N commits, and extracts
function/class definitions from Python, JS/TS, Java, and Go files.
"""

import git
import ast
import os
import re
import tempfile
import shutil
from datetime import datetime

# Supported file extensions and their parser type
SUPPORTED_EXTENSIONS = {
    '.py':   'python',
    '.js':   'js_ts',
    '.ts':   'js_ts',
    '.jsx':  'js_ts',
    '.tsx':  'js_ts',
    '.java': 'generic',
    '.go':   'generic',
}


def clone_repo(url: str, temp_dir: str) -> git.Repo:
    """Clone a public GitHub repo to a temporary directory."""
    return git.Repo.clone_from(url, temp_dir)


def extract_functions_from_python(code: str, file_name: str, commit_info: dict) -> list:
    """Extract function and class definitions from Python source code using AST."""
    chunks = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunk_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
                lines = code.split('\n')
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start + 10)
                chunk_code = '\n'.join(lines[start:end])
                if len(chunk_code.strip()) > 20:
                    chunks.append({
                        **commit_info,
                        "file_name": file_name,
                        "chunk_type": chunk_type,
                        "chunk_name": node.name,
                        "start_line": node.lineno,
                        "end_line": end,
                        "code": chunk_code
                    })
    except SyntaxError:
        pass
    except Exception:
        pass
    return chunks


def extract_functions_from_js_ts(code: str, file_name: str, commit_info: dict) -> list:
    """Extract functions from JS/TS/JSX/TSX using regex patterns."""
    chunks = []
    patterns = [
        r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(',
        r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{',
    ]
    lines = code.split('\n')
    seen_names = set()
    for i, line in enumerate(lines):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1)
                if name in seen_names or name in ('if', 'for', 'while', 'switch', 'catch'):
                    continue
                seen_names.add(name)
                end = min(i + 15, len(lines))
                chunk_code = '\n'.join(lines[i:end])
                if len(chunk_code.strip()) > 20:
                    chunks.append({
                        **commit_info,
                        "file_name": file_name,
                        "chunk_type": "function",
                        "chunk_name": name,
                        "start_line": i + 1,
                        "end_line": end,
                        "code": chunk_code
                    })
                break
    return chunks[:20]


def extract_functions_generic(code: str, file_name: str, commit_info: dict) -> list:
    """Generic extractor for Java, Go using regex."""
    chunks = []
    lines = code.split('\n')
    pattern = r'(?:public|private|protected|static|func|def)?\s*\w+\s+(\w+)\s*\('
    seen_names = set()
    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            name = match.group(1)
            if name in seen_names or name in ('if', 'for', 'while', 'switch', 'main'):
                continue
            seen_names.add(name)
            end = min(i + 15, len(lines))
            chunk_code = '\n'.join(lines[i:end])
            if len(chunk_code.strip()) > 20:
                chunks.append({
                    **commit_info,
                    "file_name": file_name,
                    "chunk_type": "function",
                    "chunk_name": name,
                    "start_line": i + 1,
                    "end_line": end,
                    "code": chunk_code
                })
    return chunks[:20]


def _get_file_extension(path: str) -> str:
    """Get the file extension, lowercased."""
    _, ext = os.path.splitext(path)
    return ext.lower()


def _extract_from_file(code: str, file_name: str, commit_info: dict) -> list:
    """Route to the correct extractor based on file extension."""
    ext = _get_file_extension(file_name)
    parser_type = SUPPORTED_EXTENSIONS.get(ext)
    if parser_type == 'python':
        return extract_functions_from_python(code, file_name, commit_info)
    elif parser_type == 'js_ts':
        return extract_functions_from_js_ts(code, file_name, commit_info)
    elif parser_type == 'generic':
        return extract_functions_generic(code, file_name, commit_info)
    return []


def _is_supported_file(path: str) -> bool:
    """Check if a file has a supported extension."""
    ext = _get_file_extension(path)
    return ext in SUPPORTED_EXTENSIONS


def walk_commits(repo_url: str, max_commits: int = 50, progress_callback=None) -> list:
    """Clone a repo and walk the last N commits, extracting code chunks.

    Supports Python, JavaScript, TypeScript, Java, and Go files.

    Args:
        repo_url: Public GitHub repository URL
        max_commits: Maximum number of commits to analyze (default 50)
        progress_callback: Optional callable(str) for status updates

    Returns:
        list[dict]: All extracted code chunks across all commits
    """
    temp_dir = tempfile.mkdtemp()
    all_chunks = []

    try:
        if progress_callback:
            progress_callback("🔍 Cloning repository...")

        repo = clone_repo(repo_url, temp_dir)
        commits = list(repo.iter_commits('HEAD', max_count=max_commits))
        commits.reverse()  # Process oldest first

        for i, commit in enumerate(commits):
            if progress_callback:
                progress_callback(f"📂 Processing commit {i + 1}/{len(commits)}: {commit.hexsha[:7]}")

            commit_info = {
                "commit_hash": commit.hexsha[:7],
                "commit_date": commit.committed_datetime.isoformat(),
                "commit_message": commit.message.strip()[:100],
                "author": str(commit.author.name),
                "commit_index": i
            }

            try:
                if commit.parents:
                    diffs = commit.diff(commit.parents[0])
                    for diff in diffs:
                        file_path = diff.b_path or diff.a_path
                        if file_path and _is_supported_file(file_path):
                            try:
                                blob = commit.tree[file_path]
                                code = blob.data_stream.read().decode('utf-8', errors='ignore')
                                chunks = _extract_from_file(code, file_path, commit_info)
                                all_chunks.extend(chunks)
                            except (KeyError, Exception):
                                pass
                else:
                    # First commit — walk ALL supported files
                    for blob in commit.tree.traverse():
                        if hasattr(blob, 'path') and _is_supported_file(blob.path):
                            try:
                                code = blob.data_stream.read().decode('utf-8', errors='ignore')
                                chunks = _extract_from_file(code, blob.path, commit_info)
                                all_chunks.extend(chunks)
                            except Exception:
                                pass
            except Exception:
                pass

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return all_chunks
