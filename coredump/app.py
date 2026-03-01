"""
CoreDump — Temporal Semantic Analysis Engine
Powered by Endee vector database.
Streamlit app with scrollable webapp layout:
  Navbar → Hero → Analyze → How It Works → Results → FAQ → Footer
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import uuid
import re
from datetime import datetime

# Local modules
from endee_client import ping, create_index, insert_vectors, generate_vector_id
from git_walker import walk_commits
from embedder import embed_batch, get_model
from analyzer import (
    semantic_drift_over_time, find_ghost_concepts,
    architecture_evolution, hottest_modules,
    build_commit_activity_data, generate_health_summary
)

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoreDump",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #F5EDE8 !important;
    color: #493129 !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

.block-container {
    padding-top: 24px !important;
    padding-left: 40px !important;
    padding-right: 40px !important;
    max-width: 1200px !important;
}

/* Cards */
.card {
    background: #FFFFFF;
    border: 1px solid #D4B8B0;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
}

.input-section {
    background: #FFFFFF;
    border: 1px solid #D4B8B0;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 32px;
}

.input-label {
    font-size: 11px;
    font-weight: 600;
    color: #8B597B;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Feature cards */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #D4B8B0;
    border-radius: 16px;
    padding: 28px;
    height: 100%;
    min-height: 180px;
}

.feature-num {
    font-size: 11px;
    font-weight: 700;
    color: #EFA3A0;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    color: #493129;
    margin-bottom: 8px;
}

.feature-desc {
    font-size: 14px;
    color: #8B597B;
    line-height: 1.6;
}

/* Stat cards */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #D4B8B0;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}

.stat-number {
    font-size: 32px;
    font-weight: 700;
    color: #493129;
    font-family: 'JetBrains Mono', monospace;
}

.stat-label {
    font-size: 13px;
    color: #8B597B;
    margin-top: 4px;
    font-weight: 500;
}

.stat-sub {
    font-size: 11px;
    color: #B09090;
    margin-top: 2px;
}

/* Ghost cards */
.ghost-card {
    background: #FFFFFF;
    border: 1px solid #D4B8B0;
    border-left: 4px solid #EFA3A0;
    border-radius: 0 14px 14px 0;
    padding: 18px 20px;
    margin-bottom: 12px;
}

.ghost-card .chunk-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    color: #493129;
    margin-bottom: 6px;
}

.ghost-card .file-path {
    color: #8B597B;
    font-size: 13px;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.ghost-card .last-seen {
    color: #B09090;
    font-size: 12px;
}

/* Badges */
.badge-high   { background: #FFE8E8; color: #C0504D; padding: 2px 10px; border-radius: 20px; font-size: 12px; display: inline-block; }
.badge-medium { background: #FFF3E0; color: #B07030; padding: 2px 10px; border-radius: 20px; font-size: 12px; display: inline-block; }
.badge-low    { background: #E8F5E8; color: #4A7A4A; padding: 2px 10px; border-radius: 20px; font-size: 12px; display: inline-block; }
.badge-type   { background: #F8DEC7; color: #493129; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; display: inline-block; }

/* Status */
.status-connected { background: #E8F5E8; color: #4A7A4A; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 500; }
.status-offline   { background: #FFE8E8; color: #C0504D; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 500; }

/* FAQ cards */
.faq-card {
    background: #FFFFFF;
    border-left: 3px solid #8B597B;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 10px;
}

.faq-q { font-weight: 600; color: #493129; font-size: 15px; margin-bottom: 6px; }
.faq-a { color: #8B597B; font-size: 14px; line-height: 1.6; }

/* Streamlit overrides */
.stTextInput input {
    background: #F5EDE8 !important;
    border: 1px solid #D4B8B0 !important;
    border-radius: 10px !important;
    color: #493129 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
}

.stTextInput input::placeholder { color: #B09090 !important; }

.stButton > button {
    background: #493129 !important;
    color: #F5EDE8 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}

.stButton > button:hover { background: #6B4035 !important; }

.stTabs [data-baseweb="tab-list"] {
    background-color: #EDE0D9 !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1px solid #D4B8B0 !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
}

.stTabs button[data-baseweb="tab"], .stTabs div[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #8B597B !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    margin: 0 !important;
    border: none !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    min-width: 80px !important;
    text-align: center !important;
    cursor: pointer !important;
}

.stTabs button[data-baseweb="tab"][aria-selected="true"], .stTabs div[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #493129 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(73,49,41,0.1) !important;
}

.stTabs button[data-baseweb="tab"]:hover, .stTabs div[data-baseweb="tab"]:hover {
    background-color: rgba(255,255,255,0.6) !important;
    color: #493129 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

.stProgress > div > div { background: #493129 !important; }
.stSlider [data-baseweb="slider"] { color: #493129 !important; }

.streamlit-expanderHeader {
    background: #FFFFFF !important;
    border: 1px solid #D4B8B0 !important;
    border-radius: 10px !important;
    color: #493129 !important;
    font-weight: 600 !important;
}

.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #8B597B;
    font-size: 16px;
}

.empty-state .empty-icon { font-size: 36px; margin-bottom: 16px; opacity: 0.4; }

/* Health circle */
.health-circle {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px auto;
    background: #FFFFFF;
    border: 3px solid;
}

.health-circle .score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
}

.health-circle .label {
    font-size: 13px;
    font-weight: 500;
    margin-top: 4px;
}

/* Section label */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #8B597B;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: #493129;
    margin-bottom: 4px;
}

.section-desc {
    font-size: 14px;
    color: #8B597B;
    margin-bottom: 20px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────
COLORS = {
    'bg': '#F5EDE8',
    'bg_section': '#EDE0D9',
    'card': '#FFFFFF',
    'border': '#D4B8B0',
    'primary': '#493129',
    'secondary': '#8B597B',
    'accent': '#EFA3A0',
    'accent_dark': '#D4827E',
    'peach': '#F8DEC7',
    'text': '#493129',
    'text_muted': '#8B597B',
    'text_light': '#B09090',
    'chart_bg': '#FFFFFF',
    'grid': '#F0E4DF',
    'success': '#5C7A5C',
}

CHART_COLORS = ['#493129', '#8B597B', '#EFA3A0', '#D4827E', '#F8DEC7']
DRIFT_LINE_COLORS = ['#493129', '#8B597B', '#EFA3A0', '#D4827E', '#8B5E7B']

QUARTILE_COLORS = {
    'Q1': '#F8DEC7',
    'Q2': '#EFA3A0',
    'Q3': '#8B597B',
    'Q4': '#493129',
}


# ─────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return get_model()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def drift_badge(score: float) -> str:
    if score > 0.7:
        return f'<span class="badge-high">{score:.3f}</span>'
    elif score > 0.3:
        return f'<span class="badge-medium">{score:.3f}</span>'
    else:
        return f'<span class="badge-low">{score:.3f}</span>'


def parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        try:
            return datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return datetime.now()


def format_date(date_str: str) -> str:
    return parse_date(date_str).strftime("%b %d")


def chart_layout(fig, height=420, showlegend=True):
    """Apply warm light theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor=COLORS['chart_bg'],
        plot_bgcolor=COLORS['chart_bg'],
        font=dict(color=COLORS['text'], family='Inter'),
        height=height,
        showlegend=showlegend,
        legend=dict(bgcolor='rgba(255,255,255,0)', font=dict(color=COLORS['text_muted'], size=11)),
        margin=dict(l=50, r=30, t=40, b=50),
        xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
    )
    return fig


def truncate_name(name, max_len=25):
    return "..." + name[-(max_len - 3):] if len(name) > max_len else name


# ═════════════════════════════════════════════════════════════
# NAVBAR
# ═════════════════════════════════════════════════════════════
endee_online = ping()

st.markdown("""
<nav style="
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    background: #F5EDE8;
    border-bottom: 1px solid #D4B8B0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    z-index: 999;
    font-family: 'Inter', sans-serif;
">
    <span style="font-weight: 700; font-size: 16px; color: #493129; letter-spacing: -0.3px;">CoreDump</span>
    <div style="display: flex; gap: 28px; align-items: center;">
        <a href="#analyze" style="color: #8B597B; text-decoration: none; font-size: 14px; font-weight: 500;">Analyze</a>
        <a href="#how-it-works" style="color: #8B597B; text-decoration: none; font-size: 14px; font-weight: 500;">How it works</a>
        <a href="#faq" style="color: #8B597B; text-decoration: none; font-size: 14px; font-weight: 500;">FAQ</a>
        <span style="background: #493129; color: #F5EDE8; padding: 6px 16px; border-radius: 8px; font-size: 13px; font-weight: 600;">Open Source</span>
    </div>
</nav>
<div style="height: 56px;"></div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SECTION 1 — HERO
# ═════════════════════════════════════════════════════════════
hero_left, hero_right = st.columns([5, 1])

with hero_left:
    st.markdown("""
    <div style="padding: 60px 0 48px 0; border-bottom: 1px solid #D4B8B0;">
        <div style="font-size: 11px; font-weight: 600; color: #8B597B; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px;">
            Powered by Endee Vector Database
        </div>
        <h1 style="font-size: 64px; font-weight: 800; color: #493129; letter-spacing: -2px; line-height: 1.05; margin: 0 0 20px 0;">
            Watch your codebase<br>evolve.
        </h1>
        <p style="font-size: 20px; color: #8B597B; max-width: 560px; line-height: 1.6; margin: 0 0 28px 0; font-weight: 400;">
            Paste any public GitHub repo URL. CoreDump analyzes semantic drift,
            ghost concepts, and architectural evolution across every commit.
        </p>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="background: #F8DEC7; color: #493129; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 500;">Semantic Analysis</span>
            <span style="background: #F8DEC7; color: #493129; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 500;">Vector Embeddings</span>
            <span style="background: #F8DEC7; color: #493129; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 500;">Git History</span>
            <span style="background: #F8DEC7; color: #493129; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 500;">Python · JS · TS · Java · Go</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    if endee_online:
        st.markdown("""
        <div style="text-align: right; padding-top: 70px;">
            <span class="status-connected">● Connected</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: right; padding-top: 70px;">
            <span class="status-offline">● Offline</span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SECTION 2 — ANALYZE (always visible)
# ═════════════════════════════════════════════════════════════
st.markdown('<div id="analyze" style="padding-top: 24px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="input-label" style="margin-bottom: 12px;">REPOSITORY URL</div>', unsafe_allow_html=True)
repo_url = st.text_input(
    "repo_url",
    placeholder="https://github.com/username/repo",
    label_visibility="collapsed",
    key="repo_url_input"
)
st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
analyze_clicked = st.button("Analyze Repository", key="analyze_btn", use_container_width=True)


# ═════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═════════════════════════════════════════════════════════════
if analyze_clicked:
    if not repo_url or not repo_url.strip():
        st.error("Please enter a GitHub repository URL.")
        st.stop()

    if not re.match(r'^https?://github\.com/[^/]+/[^/]+', repo_url.strip()):
        st.error("Please enter a valid GitHub repository URL (e.g., https://github.com/username/repo)")
        st.stop()

    if not endee_online:
        st.error("Endee is offline. Start it first:\n\n```\ndocker run -p 8080:8080 -v endee-data:/data --name endee-server endeeio/endee-server:latest\n```")
        st.stop()

    st.session_state['analysis_complete'] = False

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1 & 2: Clone + Walk
    status_text.markdown("**Step 1/5** — Cloning repository...")
    progress_bar.progress(5)

    try:
        def git_progress(msg):
            status_text.markdown(f"**Step 2/5** — {msg}")
        chunks = walk_commits(repo_url.strip(), max_commits=30, progress_callback=git_progress)
        progress_bar.progress(30)
    except Exception as e:
        st.error(f"Failed to clone repository: {str(e)}\n\nMake sure the URL is correct and the repo is public.")
        st.stop()

    if not chunks:
        st.warning("No code chunks found. CoreDump supports Python, JS, TS, Java, and Go files.")
        st.stop()

    status_text.markdown(f"**Step 2/5** — Extracted **{len(chunks)}** code chunks")
    progress_bar.progress(35)

    # Step 3: Embeddings
    status_text.markdown("**Step 3/5** — Loading embedding model...")
    progress_bar.progress(40)
    load_embedding_model()

    codes = [c['code'] for c in chunks]

    def embed_progress(current, total):
        pct = 40 + int((current / total) * 25)
        status_text.markdown(f"**Step 3/5** — Generating embeddings ({current}/{total})")
        progress_bar.progress(min(pct, 65))

    try:
        embeddings = embed_batch(codes, progress_callback=embed_progress)
    except Exception as e:
        st.error(f"Embedding generation failed: {str(e)}")
        st.stop()

    progress_bar.progress(65)

    # Step 4: Store in Endee
    status_text.markdown("**Step 4/5** — Storing vectors in Endee...")
    progress_bar.progress(70)

    repo_name = repo_url.strip().rstrip('/').split('/')[-1]
    index_name = f"coredump_{repo_name}_{uuid.uuid4().hex[:8]}"

    try:
        create_index(index_name, dimension=len(embeddings[0]))
        vectors_to_insert = []
        for chunk, emb in zip(chunks, embeddings):
            vectors_to_insert.append({
                "id": generate_vector_id(),
                "vector": emb,
                "metadata": {
                    "commit_hash": chunk['commit_hash'],
                    "commit_date": chunk['commit_date'],
                    "commit_message": chunk['commit_message'],
                    "author": chunk['author'],
                    "file_name": chunk['file_name'],
                    "chunk_name": chunk['chunk_name'],
                    "chunk_type": chunk['chunk_type'],
                    "commit_index": chunk['commit_index']
                }
            })
        insert_vectors(index_name, vectors_to_insert)
        progress_bar.progress(80)
        status_text.markdown(f"**Step 4/5** — Stored **{len(vectors_to_insert)}** vectors in Endee")
    except Exception as e:
        st.warning(f"Endee storage issue: {str(e)} — Continuing with local analysis.")

    # Step 5: Analysis
    status_text.markdown("**Step 5/5** — Running semantic analysis...")
    progress_bar.progress(85)

    try:
        drift_data = semantic_drift_over_time(chunks, embeddings)
        ghost_data = find_ghost_concepts(chunks)
        arch_data = architecture_evolution(chunks, embeddings)
        heat_data = hottest_modules(drift_data)
        activity_data = build_commit_activity_data(chunks)
        summary_data = generate_health_summary(drift_data, ghost_data, heat_data, chunks)
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.stop()

    progress_bar.progress(100)
    status_text.markdown("**Analysis complete**")

    st.session_state['analysis_complete'] = True
    st.session_state['chunks'] = chunks
    st.session_state['embeddings'] = embeddings
    st.session_state['drift_data'] = drift_data
    st.session_state['ghost_data'] = ghost_data
    st.session_state['arch_data'] = arch_data
    st.session_state['heat_data'] = heat_data
    st.session_state['activity_data'] = activity_data
    st.session_state['summary_data'] = summary_data
    st.session_state['repo_url'] = repo_url
    st.session_state['index_name'] = index_name


# ═════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS (only after analysis)
# ═════════════════════════════════════════════════════════════
if st.session_state.get('analysis_complete', False):
    chunks = st.session_state['chunks']
    embeddings = st.session_state['embeddings']
    drift_data = st.session_state['drift_data']
    ghost_data = st.session_state['ghost_data']
    arch_data = st.session_state['arch_data']
    heat_data = st.session_state['heat_data']
    activity_data = st.session_state.get('activity_data')
    summary_data = st.session_state.get('summary_data')

    st.markdown("""
    <div style="padding: 48px 0 24px 0; border-top: 1px solid #D4B8B0;">
        <div class="section-label">Analysis Results</div>
        <h2 style="font-size: 32px; font-weight: 700; color: #493129; margin: 0 0 32px 0;">Your codebase, dissected.</h2>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Drift Wave",
        "Ghost Concepts",
        "Architecture Map",
        "Activity",
        "Module Heat",
        "Summary"
    ])

    # ─── TAB 1: Drift Wave ───
    with tab1:
        st.markdown('<div class="section-heading">Semantic Drift Over Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">How each module\'s meaning shifted across commits.</div>', unsafe_allow_html=True)

        if not drift_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">~</div>No drift data available.</div>', unsafe_allow_html=True)
        else:
            file_avg_drift = {}
            for fname, commits in drift_data.items():
                scores = [c['drift_score'] for c in commits]
                file_avg_drift[fname] = np.mean(scores) if scores else 0

            top_files = sorted(file_avg_drift.keys(), key=lambda f: file_avg_drift[f], reverse=True)[:5]
            fig = go.Figure()

            for idx, fname in enumerate(top_files):
                file_commits = drift_data[fname]
                dates = [format_date(c['commit_date']) for c in file_commits]
                scores = [c['drift_score'] for c in file_commits]
                hashes = [c['commit_hash'] for c in file_commits]
                messages = [c['commit_message'] for c in file_commits]
                hover_text = [
                    f"<b>{fname}</b><br>Commit: {h}<br>Message: {m[:50]}<br>Drift: {s:.3f}"
                    for h, m, s in zip(hashes, messages, scores)
                ]
                display_name = fname if len(fname) < 40 else "..." + fname[-37:]
                fig.add_trace(go.Scatter(
                    x=dates, y=scores, name=display_name, mode='lines+markers',
                    line=dict(width=2.5, color=DRIFT_LINE_COLORS[idx % len(DRIFT_LINE_COLORS)]),
                    marker=dict(size=6, color=DRIFT_LINE_COLORS[idx % len(DRIFT_LINE_COLORS)]),
                    hovertext=hover_text, hoverinfo='text'
                ))

            fig.add_hline(y=0.3, line_dash="dash", line_color=COLORS['text_light'],
                         annotation_text="Low", annotation_position="top left",
                         annotation_font_color=COLORS['text_light'])
            fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS['accent_dark'],
                         annotation_text="High", annotation_position="top left",
                         annotation_font_color=COLORS['accent_dark'])

            fig.update_layout(xaxis_title="Commit Date", yaxis_title="Drift Score", yaxis=dict(range=[0, 1]))
            chart_layout(fig, height=420)
            st.plotly_chart(fig, use_container_width=True, key="drift_chart")

            # Drift table
            st.markdown('<div class="section-heading" style="font-size: 16px; margin-top: 20px;">Most Drifted Commits</div>', unsafe_allow_html=True)
            all_drift_rows = []
            for fname, commits in drift_data.items():
                for c in commits:
                    all_drift_rows.append({'Commit': c['commit_hash'], 'Message': c['commit_message'][:60],
                                           'Date': format_date(c['commit_date']), 'Drift Score': c['drift_score'], 'Module': fname})
            if all_drift_rows:
                df_drift = pd.DataFrame(all_drift_rows).sort_values('Drift Score', ascending=False).head(10).reset_index(drop=True)
                tbl = '<div class="card" style="overflow-x:auto; padding:0;"><table style="width:100%;border-collapse:collapse;font-size:13px;">'
                tbl += '<tr style="border-bottom:1px solid #D4B8B0;">'
                for col in ['Commit', 'Message', 'Date', 'Drift Score', 'Module']:
                    tbl += f'<th style="padding:14px 12px;text-align:left;color:#8B597B;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">{col}</th>'
                tbl += '</tr>'
                for _, row in df_drift.iterrows():
                    tbl += '<tr style="border-bottom:1px solid #F0E4DF;">'
                    tbl += f'<td style="padding:12px;font-family:JetBrains Mono,monospace;font-size:12px;color:#493129;">{row["Commit"]}</td>'
                    tbl += f'<td style="padding:12px;color:#493129;">{row["Message"]}</td>'
                    tbl += f'<td style="padding:12px;color:#8B597B;">{row["Date"]}</td>'
                    tbl += f'<td style="padding:12px;">{drift_badge(row["Drift Score"])}</td>'
                    mod = row["Module"] if len(row["Module"]) < 35 else "..." + row["Module"][-32:]
                    tbl += f'<td style="padding:12px;color:#8B597B;font-size:12px;">{mod}</td>'
                    tbl += '</tr>'
                tbl += '</table></div>'
                st.markdown(tbl, unsafe_allow_html=True)

    # ─── TAB 2: Ghost Concepts ───
    with tab2:
        st.markdown('<div class="section-heading">Ghost Concepts</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Code that once lived here — and quietly disappeared.</div>', unsafe_allow_html=True)

        if not ghost_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">○</div>No ghost concepts found — your codebase has perfect memory.</div>', unsafe_allow_html=True)
        else:
            cols = st.columns(2)
            for i, ghost in enumerate(ghost_data):
                with cols[i % 2]:
                    last_seen_fmt = format_date(ghost['last_seen_date'])
                    file_display = ghost['file_name'] if len(ghost['file_name']) <= 45 else "..." + ghost['file_name'][-42:]
                    st.markdown(f"""
                    <div class="ghost-card">
                        <div class="chunk-name">{ghost['chunk_name']}</div>
                        <span class="badge-type">{ghost['chunk_type']}</span>
                        <div class="file-path">{file_display}</div>
                        <div class="last-seen">Last seen: {last_seen_fmt} · commit <code style="color:#8B597B;font-family:JetBrains Mono,monospace;font-size:11px;">{ghost['last_seen_hash']}</code></div>
                    </div>
                    """, unsafe_allow_html=True)

    # ─── TAB 3: Architecture Map ───
    with tab3:
        st.markdown('<div class="section-heading">Architecture Map</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">How your codebase drifted through vector space over time.</div>', unsafe_allow_html=True)

        if not arch_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">◇</div>Not enough data points.</div>', unsafe_allow_html=True)
        else:
            max_ci = max(d['commit_index'] for d in arch_data)
            time_slider = st.slider("Showing commits up to:", min_value=0, max_value=max_ci, value=max_ci, step=1, key="arch_time_slider")
            filtered_arch = [d for d in arch_data if d['commit_index'] <= time_slider]

            if filtered_arch:
                def get_q_color(ci):
                    if max_ci == 0: return QUARTILE_COLORS['Q1']
                    r = ci / max_ci
                    if r <= 0.25: return QUARTILE_COLORS['Q1']
                    elif r <= 0.50: return QUARTILE_COLORS['Q2']
                    elif r <= 0.75: return QUARTILE_COLORS['Q3']
                    return QUARTILE_COLORS['Q4']

                def get_q_label(ci):
                    if max_ci == 0: return "Q1 — Oldest"
                    r = ci / max_ci
                    if r <= 0.25: return "Q1 — Oldest"
                    elif r <= 0.50: return "Q2"
                    elif r <= 0.75: return "Q3"
                    return "Q4 — Newest"

                q_groups = {}
                for d in filtered_arch:
                    ql = get_q_label(d['commit_index'])
                    if ql not in q_groups:
                        q_groups[ql] = {'x': [], 'y': [], 'hover': [], 'color': get_q_color(d['commit_index'])}
                    q_groups[ql]['x'].append(d['x'])
                    q_groups[ql]['y'].append(d['y'])
                    q_groups[ql]['hover'].append(
                        f"<b>{d['chunk_name']}</b><br>File: {d['file_name']}<br>Author: {d['author']}<br>Date: {format_date(d['commit_date'])}<br>Commit: {d['commit_hash']}"
                    )

                fig = go.Figure()
                for ql in ["Q1 — Oldest", "Q2", "Q3", "Q4 — Newest"]:
                    if ql in q_groups:
                        grp = q_groups[ql]
                        fig.add_trace(go.Scatter(
                            x=grp['x'], y=grp['y'], mode='markers', name=ql,
                            marker=dict(size=8, color=grp['color'], opacity=0.8, line=dict(width=0.5, color=COLORS['border'])),
                            hovertext=grp['hover'], hoverinfo='text'
                        ))

                fig.update_layout(xaxis_title="UMAP 1", yaxis_title="UMAP 2",
                                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), legend_title="Timeline")
                chart_layout(fig, height=500)
                st.plotly_chart(fig, use_container_width=True, key="arch_chart")

    # ─── TAB 4: Activity ───
    with tab4:
        st.markdown('<div class="section-heading">Commit Activity</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Development activity across the analyzed history.</div>', unsafe_allow_html=True)

        if not activity_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">▦</div>Not enough commit data.</div>', unsafe_allow_html=True)
        else:
            colorscale = [[0.0, '#F5EDE8'], [0.25, '#F8DEC7'], [0.5, '#EFA3A0'], [0.75, '#D4827E'], [1.0, '#493129']]
            fig = go.Figure(data=go.Heatmap(
                z=activity_data['grid_z'], x=activity_data['x_labels'], y=activity_data['y_labels'],
                text=activity_data['grid_hover'], hoverinfo='text', colorscale=colorscale,
                showscale=False, xgap=3, ygap=3
            ))
            fig.update_layout(yaxis=dict(autorange='reversed'), xaxis=dict(side='top', tickangle=-45, tickfont=dict(size=10)))
            chart_layout(fig, height=220, showlegend=False)
            fig.update_layout(margin=dict(l=60, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True, key="activity_chart")

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            ma = activity_data['most_active_day']
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{ma["count"]}</div><div class="stat-label">Most Active Day</div><div class="stat-sub">{ma["date"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{activity_data["longest_streak"]}</div><div class="stat-label">Longest Streak</div><div class="stat-sub">consecutive days</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{activity_data["total_active_days"]}</div><div class="stat-label">Active Days</div><div class="stat-sub">total</div></div>', unsafe_allow_html=True)

    # ─── TAB 5: Module Heat ───
    with tab5:
        st.markdown('<div class="section-heading">Module Volatility</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Files ranked by semantic drift — higher means more volatile.</div>', unsafe_allow_html=True)

        if not heat_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">□</div>No module heat data.</div>', unsafe_allow_html=True)
        else:
            file_names = [h['file_name'] for h in heat_data]
            avg_drifts = [h['avg_drift'] for h in heat_data]
            max_drifts = [h['max_drift'] for h in heat_data]
            total_c_list = [h['total_commits'] for h in heat_data]
            bar_colors = ['#C0504D' if d > 0.7 else '#B07030' if d > 0.3 else '#5C7A5C' for d in avg_drifts]
            display_names = [f if len(f) < 50 else "..." + f[-47:] for f in file_names]
            hover_text = [f"<b>{f}</b><br>Avg: {a:.3f}<br>Max: {m:.3f}<br>Commits: {t}" for f, a, m, t in zip(file_names, avg_drifts, max_drifts, total_c_list)]

            fig = go.Figure()
            fig.add_trace(go.Bar(y=display_names, x=avg_drifts, orientation='h',
                                marker=dict(color=bar_colors, line=dict(width=0)),
                                hovertext=hover_text, hoverinfo='text'))
            fig.update_layout(xaxis_title="Average Drift Score", xaxis=dict(range=[0, 1]),
                            yaxis=dict(autorange="reversed"), showlegend=False)
            chart_layout(fig, height=max(400, len(file_names) * 30), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="heat_chart")

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            ms = min(heat_data, key=lambda x: x['avg_drift'])
            mv = max(heat_data, key=lambda x: x['avg_drift'])
            tc = len(set(c['commit_hash'] for c in chunks))
            gc = len(ghost_data)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:14px;">{truncate_name(ms["file_name"])}</div><div class="stat-label">Most Stable</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:14px;">{truncate_name(mv["file_name"])}</div><div class="stat-label">Most Volatile</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{tc}</div><div class="stat-label">Commits</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{gc}</div><div class="stat-label">Ghosts</div></div>', unsafe_allow_html=True)

    # ─── TAB 6: Summary ───
    with tab6:
        st.markdown('<div class="section-heading">Health Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Automated assessment of codebase stability.</div>', unsafe_allow_html=True)

        if not summary_data:
            st.markdown('<div class="empty-state"><div class="empty-icon">—</div>Insufficient data — try a larger repo.</div>', unsafe_allow_html=True)
        else:
            score = summary_data['health_score']
            verdict = summary_data['verdict']
            v_desc = summary_data['verdict_desc']
            v_color = '#5C7A5C' if score >= 80 else '#B07030' if score >= 60 else '#C0504D'

            st.markdown(f"""
            <div style="text-align: center; padding: 30px 0 10px 0;">
                <div class="health-circle" style="border-color: {v_color};">
                    <span class="score" style="color: {v_color};">{score}</span>
                    <span class="label" style="color: {v_color};">{verdict}</span>
                </div>
                <p style="color: #8B597B; font-size: 14px; max-width: 500px; margin: 0 auto;">{v_desc}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{summary_data["total_functions"]}</div><div class="stat-label">Functions Analyzed</div><div class="stat-sub">across {summary_data["total_files"]} files</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{summary_data["ghost_count"]}</div><div class="stat-label">Ghost Concepts</div><div class="stat-sub">abandoned features</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:14px;">{truncate_name(summary_data["most_volatile"], 22)}</div><div class="stat-label">Most Volatile</div><div class="stat-sub">highest drift</div></div>', unsafe_allow_html=True)

            st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-heading" style="font-size:16px;">Insights</div>', unsafe_allow_html=True)
            ins_html = '<div class="card" style="padding:0;">'
            for j, insight in enumerate(summary_data['insights']):
                bdr = 'border-bottom:1px solid #F0E4DF;' if j < len(summary_data['insights']) - 1 else ''
                ins_html += f'<div style="padding:14px 24px;{bdr}color:#493129;font-size:14px;line-height:1.7;">▸ {insight}</div>'
            ins_html += '</div>'
            st.markdown(ins_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SECTION 3 — HOW IT WORKS (always visible)
# ═════════════════════════════════════════════════════════════
st.markdown('<div id="how-it-works" style="height: 1px;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="padding: 60px 0 24px 0; border-top: 1px solid #D4B8B0;">
    <div class="section-label">How it works</div>
    <h2 style="font-size: 32px; font-weight: 700; color: #493129; margin: 0 0 40px 0;">Four lenses on your codebase.</h2>
</div>
""", unsafe_allow_html=True)

fc1, fc2 = st.columns(2)
with fc1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-num">01</div>
        <div class="feature-title">Semantic Drift Wave</div>
        <div class="feature-desc">Track how each module's meaning shifts commit by commit. Catch the exact moment things started going wrong.</div>
    </div>
    """, unsafe_allow_html=True)
with fc2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-num">02</div>
        <div class="feature-title">Ghost Concepts</div>
        <div class="feature-desc">Find functions that quietly disappeared from your codebase. Institutional knowledge that died with a git push.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

fc3, fc4 = st.columns(2)
with fc3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-num">03</div>
        <div class="feature-title">Architecture Map</div>
        <div class="feature-desc">2D UMAP projection of your entire codebase evolving over time. Watch clusters form, split, and drift.</div>
    </div>
    """, unsafe_allow_html=True)
with fc4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-num">04</div>
        <div class="feature-title">Health Summary</div>
        <div class="feature-desc">Plain English report on codebase health, stability score, and actionable insights. No AI hype — just data.</div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SECTION 5 — FAQ (always visible)
# ═════════════════════════════════════════════════════════════
st.markdown('<div id="faq" style="height: 1px;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="padding: 60px 0 24px 0; border-top: 1px solid #D4B8B0;">
    <div class="section-label">FAQ</div>
    <h2 style="font-size: 32px; font-weight: 700; color: #493129; margin: 0 0 32px 0;">Common questions.</h2>
</div>
""", unsafe_allow_html=True)

faq_items = [
    ("What is CoreDump?",
     "CoreDump is a temporal semantic analysis engine. Unlike static code analyzers "
     "that show what your codebase looks like today, CoreDump tracks how the meaning "
     "and architecture of your code has evolved across every commit in its history."),
    ("What are Ghost Concepts?",
     "Ghost concepts are functions or classes that once existed in your codebase "
     "but disappeared across commits. They represent abandoned features, deleted "
     "experiments, or lost institutional knowledge."),
    ("What is Semantic Drift?",
     "Drift measures how much the meaning of a module changes over time. "
     "A score near 0 means architecturally consistent. Near 1 means "
     "the module has fundamentally changed what it does."),
    ("Why Endee as the vector database?",
     "CoreDump generates one embedding per code chunk per commit. "
     "A medium repo with 50 commits and 300 functions = 15,000 vectors. "
     "Endee's HNSW indexing handles similarity search across all of "
     "these in milliseconds. No traditional database could do this."),
    ("Which languages are supported?",
     "Python, JavaScript, TypeScript, Java, and Go."),
    ("Why is my repo showing no chunks?",
     "CoreDump analyzes function-level code. Try a larger public repo "
     "like github.com/psf/requests or github.com/pallets/flask."),
]

for q, a in faq_items:
    with st.expander(q, expanded=False):
        st.markdown(f'<div style="color: #8B597B; font-size: 14px; line-height: 1.6;">{a}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SECTION 6 — FOOTER (always visible)
# ═════════════════════════════════════════════════════════════
st.markdown("""
<footer style="
    text-align: center;
    padding: 32px 0;
    border-top: 1px solid #D4B8B0;
    color: #B09090;
    font-size: 13px;
    margin-top: 40px;
">
    Built with <a href="https://github.com/endee-io/endee" style="color: #8B597B; text-decoration: none; font-weight: 500;">Endee Vector Database</a>
    &nbsp;·&nbsp; Powered by CodeBERT
    &nbsp;·&nbsp; CoreDump
</footer>
""", unsafe_allow_html=True)
