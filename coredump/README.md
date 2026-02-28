# CoreDump

**Watch your codebase evolve. Commit by commit.**

*A Temporal Semantic Analysis Engine powered by Endee Vector Database.*

🌐 **Live Demo:** https://coredump.onrender.com

---

## The Problem

Static code analyzers tell you what your code looks like **today**.

But codebases are living things. Over hundreds of commits, the original intent of a module slowly drifts. Functions die out and become "ghosts." Core architectures morph from focused logic into tangled god classes. Institutional knowledge disappears with a single `git push`.

Traditional tools — linters, static analyzers, even code review — are blind to this kind of **temporal decay**. They see a snapshot. CoreDump sees the whole story.

---

## What is CoreDump?

CoreDump is a temporal semantic analysis engine. Paste any public GitHub repository URL and CoreDump will:

1. Clone the repository and walk through the last N commits
2. Extract every function and class definition across Python, JS, TS, Java, and Go
3. Generate 384-dimensional semantic embeddings using CodeBERT
4. Store everything in **Endee Vector Database**
5. Analyze how the codebase's meaning has evolved over time

The result is four interactive lenses on your codebase's evolution — drift, decay, architecture, and health.

---

## Four Lenses

### Drift Wave
Track how each module's semantic meaning shifts commit by commit. Pinpoint the exact commit where things started going wrong. Drift score near 0 means stable. Near 1 means the module has fundamentally changed what it does.

### Ghost Concepts
Find functions and classes that once existed in your codebase but quietly disappeared across commits. These represent abandoned features, deleted experiments, and lost institutional knowledge — code that died with a git push.

### Architecture Map
A 2D UMAP projection of your entire codebase evolving over time. Watch semantic clusters form, split, and drift as your team evolves the system. Color-coded by commit recency — from oldest to newest.

### Health Summary
A plain English report on overall codebase health. Includes a stability score (0-100), verdict (Healthy / Aging / Decaying), and actionable insights about the most volatile and most stable modules.

---

## System Architecture

```
GitHub URL
    ↓
git_walker.py — Clone repo, walk N commits, extract code chunks
    ↓
Python AST + RegEx — Parse functions/classes per language
    ↓
embedder.py — Generate 384-dim embeddings via all-MiniLM-L6-v2
    ↓
endee_client.py — Store vectors + metadata in Endee Vector DB
    ↓
analyzer.py — Compute drift, ghost concepts, UMAP projection
    ↓
app.py — Streamlit dashboard with Plotly visualizations
```

---

## How Endee is Used

Endee is the core semantic engine of CoreDump. Without it, comparing the semantic shift of thousands of functions across dozens of commits would require an O(N²) matrix multiplication nightmare in memory.

**Storage at Scale**
CoreDump generates 10,000 to 20,000 vectors for a medium-sized repository history. Each vector is 384 dimensions with rich metadata — commit hash, author, date, file path, function name, line numbers. Endee stores and indexes all of this instantly via its REST API.

**HNSW Indexing**
Endee uses HNSW graph indexing to place every historical function into organized semantic space. This makes similarity search across tens of thousands of historical vectors happen in milliseconds — no traditional database could do this.

**Drift Calculation**
CoreDump pulls embeddings from Endee and computes `1.0 - cosine_similarity()` between consecutive versions of the same function across commits. The larger the distance, the more that module has semantically drifted from its original purpose.

**Ghost Detection**
Ghost concepts are identified by finding vectors in Endee that stop appearing after a certain commit index. CoreDump queries the timeline of each chunk name and surfaces the ones that vanished — representing dead or deleted institutional knowledge.

---

## Supported Languages

| Language | Parser |
|----------|--------|
| Python | Python AST (built-in) |
| JavaScript | RegEx function extractor |
| TypeScript | RegEx function extractor |
| Java | RegEx method extractor |
| Go | RegEx function extractor |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Database | Endee (endee-io/endee) |
| Embeddings | all-MiniLM-L6-v2 via sentence-transformers |
| Dimensionality Reduction | UMAP |
| Git Analysis | GitPython |
| Visualizations | Plotly |
| UI Framework | Streamlit |
| Language | Python 3.11+ |

---

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- Docker
- Git

### 1. Clone this repository
```bash
git clone https://github.com/Chaplot05/endee.git
cd endee
git checkout riddhi-ai-project
cd coredump
```

### 2. Start Endee Vector Database
```bash
docker run -p 8080:8080 -v endee-data:/data --name endee-server endeeio/endee-server:latest
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run CoreDump
```bash
streamlit run app.py
```

### 5. Analyze a repository
Open `http://localhost:8501`, paste any public GitHub URL, and click **Analyze Repository**.

**Recommended repos to try:**
- `https://github.com/psf/requests`
- `https://github.com/pallets/flask`
- `https://github.com/tiangolo/fastapi`

---

## Project Structure

```
coredump/
├── app.py              # Streamlit UI — dashboard and visualizations
├── endee_client.py     # Endee REST API wrapper
├── git_walker.py       # Git history walker and code parser
├── embedder.py         # CodeBERT embedding engine
├── embed_worker.py     # Subprocess worker for safe embedding
├── analyzer.py         # Drift, ghost, UMAP, and health analysis
├── requirements.txt    # Python dependencies
├── Procfile            # Render deployment config
└── README.md           # You are here
```

---

## Mandatory Repository Steps Completed

- Starred the official Endee repository at github.com/endee-io/endee
- Forked the repository to personal GitHub account (Chaplot05/endee)
- Built the project inside the forked repository on branch `riddhi-ai-project`

---

## Live Deployment

| Service | Platform | URL |
|---------|----------|-----|
| CoreDump App | Render | https://coredump.onrender.com |
| Endee Vector DB | Railway | https://endee-server-production-5a34.up.railway.app |

---

*Built with Endee Vector Database · Powered by sentence-transformers · CoreDump*
