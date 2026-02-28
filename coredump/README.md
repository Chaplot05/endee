# CoreDump

> Temporal semantic analysis engine — track how your codebase's meaning evolves across git history using vector embeddings.

---

## The Problem

Static code analyzers tell you what your code looks like **today**.

But codebases are living things — they **drift**, **decay**, and **forget**.

Functions that once existed vanish silently. Modules that were stable start shifting in meaning. Architecture that was once coherent begins to fracture across commits.

**CoreDump** tracks the semantic evolution of your codebase across git history using vector embeddings stored in the [Endee](https://github.com/endee-io/endee) vector database.

---

## What CoreDump Does

Six analyses no other tool does together:

### 🌊 Drift Wave
Track how each module's **semantic meaning** shifts over time.  
Catch the **exact commit** where things started going wrong.

### 👻 Ghost Concepts
Find functions and classes that once existed — and **quietly disappeared**.  
Institutional knowledge that died with a `git push`.

### 🗺️ Architecture Map
2D UMAP projection of your **entire codebase across time**.  
Watch clusters form, split, and drift as your team evolves the system.

### 📅 Activity Heatmap
GitHub-style **commit activity visualization** showing development patterns,
streaks, and busy periods across the analyzed history.

### 🔥 Module Heatmap
Rank every file by **semantic volatility**.  
Know which parts of your codebase are stable vs chaotic.

### 📋 Health Summary
Rule-based **codebase health report** with a 0–100 score,
verdicts (Healthy / Aging / Decaying), and auto-generated insights.

---

## Supported Languages

| Language | Parser |
|---|---|
| **Python** | Full AST parsing (functions, classes) |
| **JavaScript** | Regex-based function extraction |
| **TypeScript** | Regex-based function extraction |
| **JSX / TSX** | Regex-based function extraction |
| **Java** | Regex-based method extraction |
| **Go** | Regex-based function extraction |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CoreDump Pipeline                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   GitHub URL                                                         │
│       │                                                              │
│       ▼                                                              │
│   ┌──────────┐    ┌──────────────┐    ┌───────────────────┐         │
│   │ GitPython │───▶│  AST/Regex   │───▶│  ONNX Embedder    │         │
│   │  (clone)  │    │  Parser      │    │  (384-dim vectors)│         │
│   └──────────┘    └──────────────┘    └─────────┬─────────┘         │
│                                                  │                    │
│                                                  ▼                    │
│                                         ┌──────────────┐             │
│                                         │  Endee Vector │             │
│                                         │   Database    │             │
│                                         │  (HNSW index) │             │
│                                         └───────┬──────┘             │
│                                                  │                    │
│                                                  ▼                    │
│                                         ┌──────────────┐             │
│                                         │   Analyzer    │             │
│                                         │ (6 analyses)  │             │
│                                         └───────┬──────┘             │
│                                                  │                    │
│                                                  ▼                    │
│                                         ┌──────────────┐             │
│                                         │ Streamlit UI  │             │
│                                         │  (6 viz tabs) │             │
│                                         └──────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## How Endee Is Used

CoreDump stores **one vector per code chunk per commit** in Endee.

For a repository with 50 commits and 200 functions, that's up to **10,000 vectors** — each carrying:
- A 384-dimensional semantic embedding capturing the chunk's meaning
- Rich metadata: commit hash, date, author, file path, function/class name

Endee's high-performance **HNSW indexing** makes similarity search across all vectors instant.

---

## Setup

### 1. Start Endee

```bash
docker run -p 8080:8080 -v endee-data:/data --name endee-server endeeio/endee-server:latest
```

### 2. Install Dependencies

```bash
cd coredump
pip install -r requirements.txt
```

### 3. Run CoreDump

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) |
| **Embeddings** | all-MiniLM-L6-v2 (ONNX Runtime) |
| **Dimensionality Reduction** | UMAP |
| **UI** | Streamlit |
| **Git Analysis** | GitPython |
| **Visualizations** | Plotly |
| **Language** | Python 3.11+ |

---

## File Structure

```
coredump/
├── app.py                 # Main Streamlit application (6 tabs + FAQ)
├── endee_client.py        # Endee REST API wrapper
├── git_walker.py          # Clone repo + walk commits (multi-language)
├── embedder.py            # ONNX Runtime embeddings (torch-free)
├── analyzer.py            # 6 analysis functions
├── requirements.txt       # All dependencies
├── Procfile               # For Railway deployment
├── .env.example           # Example env file
└── README.md              # This file
```

---

## Demo Repos to Try

- **Flask** — https://github.com/pallets/flask
- **Requests** — https://github.com/psf/requests
- **FastAPI** — https://github.com/tiangolo/fastapi

---

## License

MIT

---

<div align="center">
<strong>CoreDump</strong><br>
<em>Built with Endee Vector Database · Powered by CodeBERT</em>
</div>
