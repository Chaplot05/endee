"""
CoreDump Analysis Engine
Six temporal semantic analysis functions:
1. Semantic Drift Over Time — per-file embedding drift across commits
2. Ghost Concepts — functions/classes that disappeared from the codebase
3. Architecture Evolution — 2D UMAP projection of the entire codebase over time
4. Commit Activity Heatmap — GitHub-style contribution grid
5. Module Heatmap — files ranked by semantic volatility
6. Health Summary — rule-based codebase health report
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime, timedelta
import umap


def compute_drift(embeddings: list) -> list:
    """Compute cosine distance between consecutive embeddings."""
    drifts = [0.0]
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        drift = 1.0 - sim
        drifts.append(float(max(0.0, drift)))
    return drifts


def semantic_drift_over_time(chunks: list, embeddings: list) -> dict:
    """Compute per-file semantic drift across commits."""
    file_data = {}
    for chunk, emb in zip(chunks, embeddings):
        fname = chunk['file_name']
        if fname not in file_data:
            file_data[fname] = []
        file_data[fname].append({
            'commit_index': chunk['commit_index'],
            'commit_date': chunk['commit_date'],
            'commit_hash': chunk['commit_hash'],
            'commit_message': chunk['commit_message'],
            'embedding': emb
        })

    result = {}
    for fname, data in file_data.items():
        if len(data) < 2:
            continue
        data.sort(key=lambda x: x['commit_index'])
        commit_embs = {}
        for d in data:
            ci = d['commit_index']
            if ci not in commit_embs:
                commit_embs[ci] = {'embeddings': [], 'meta': d}
            commit_embs[ci]['embeddings'].append(d['embedding'])

        sorted_commits = sorted(commit_embs.keys())
        avg_embs = []
        metas = []
        for ci in sorted_commits:
            avg = np.mean(commit_embs[ci]['embeddings'], axis=0).tolist()
            avg_embs.append(avg)
            metas.append(commit_embs[ci]['meta'])

        drifts = compute_drift(avg_embs)
        result[fname] = [
            {
                'commit_date': metas[i]['commit_date'],
                'commit_hash': metas[i]['commit_hash'],
                'commit_message': metas[i]['commit_message'],
                'drift_score': drifts[i]
            }
            for i in range(len(metas))
        ]
    return result


def find_ghost_concepts(chunks: list) -> list:
    """Find functions/classes that appeared early but vanished in recent commits."""
    if not chunks:
        return []
    max_commit = max(c['commit_index'] for c in chunks)
    if max_commit == 0:
        return []
    recent_threshold = max_commit * 0.7

    concept_timeline = {}
    for chunk in chunks:
        name = chunk['chunk_name']
        key = f"{chunk['file_name']}::{name}"
        if key not in concept_timeline:
            concept_timeline[key] = {
                'chunk_name': name,
                'first_seen': chunk['commit_index'],
                'last_seen': chunk['commit_index'],
                'last_seen_date': chunk['commit_date'],
                'last_seen_hash': chunk['commit_hash'],
                'file_name': chunk['file_name'],
                'chunk_type': chunk['chunk_type']
            }
        else:
            if chunk['commit_index'] > concept_timeline[key]['last_seen']:
                concept_timeline[key]['last_seen'] = chunk['commit_index']
                concept_timeline[key]['last_seen_date'] = chunk['commit_date']
                concept_timeline[key]['last_seen_hash'] = chunk['commit_hash']
            if chunk['commit_index'] < concept_timeline[key]['first_seen']:
                concept_timeline[key]['first_seen'] = chunk['commit_index']

    ghosts = []
    for key, timeline in concept_timeline.items():
        if timeline['last_seen'] < recent_threshold and timeline['first_seen'] < recent_threshold * 0.5:
            ghosts.append({
                'chunk_name': timeline['chunk_name'],
                'file_name': timeline['file_name'],
                'chunk_type': timeline['chunk_type'],
                'last_seen_date': timeline['last_seen_date'],
                'last_seen_hash': timeline['last_seen_hash']
            })
    ghosts.sort(key=lambda x: x['last_seen_date'], reverse=True)
    return ghosts


def architecture_evolution(chunks: list, embeddings: list) -> list:
    """Project all code chunk embeddings into 2D space using UMAP."""
    if len(embeddings) < 5:
        return []
    emb_array = np.array(embeddings)
    n_neighbors = min(15, len(embeddings) - 1)
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=0.1, metric='cosine', random_state=42
    )
    coords = reducer.fit_transform(emb_array)
    result = []
    for i, (chunk, coord) in enumerate(zip(chunks, coords)):
        result.append({
            'x': float(coord[0]),
            'y': float(coord[1]),
            'commit_date': chunk['commit_date'],
            'commit_index': chunk['commit_index'],
            'commit_hash': chunk['commit_hash'],
            'chunk_name': chunk['chunk_name'],
            'file_name': chunk['file_name'],
            'author': chunk['author'],
            'chunk_type': chunk['chunk_type']
        })
    return result


def hottest_modules(drift_data: dict) -> list:
    """Rank files by average semantic drift score."""
    result = []
    for fname, commits in drift_data.items():
        scores = [c['drift_score'] for c in commits]
        if scores:
            result.append({
                'file_name': fname,
                'avg_drift': float(np.mean(scores)),
                'max_drift': float(np.max(scores)),
                'total_commits': len(scores)
            })
    result.sort(key=lambda x: x['avg_drift'], reverse=True)
    return result


def build_commit_activity_data(chunks: list) -> dict:
    """Build data for a GitHub-style commit activity heatmap.

    Returns:
        dict with keys:
        - 'dates': list of date strings
        - 'counts': list of commit counts per date
        - 'grid': dict with 'z' (7 x weeks), 'x_labels' (week dates), 'y_labels' (day names)
        - 'most_active_day': dict with 'date' and 'count'
        - 'longest_streak': int
        - 'total_active_days': int
    """
    if not chunks:
        return None

    # Count unique commits per date
    commit_dates = defaultdict(set)
    for chunk in chunks:
        date_str = chunk['commit_date'][:10]  # YYYY-MM-DD
        commit_dates[date_str].add(chunk['commit_hash'])

    date_counts = {d: len(hashes) for d, hashes in commit_dates.items()}

    if not date_counts:
        return None

    # Parse all dates
    all_dates = sorted(date_counts.keys())
    start_date = datetime.strptime(all_dates[0], "%Y-%m-%d")
    end_date = datetime.strptime(all_dates[-1], "%Y-%m-%d")

    # Align start to Monday
    start_date = start_date - timedelta(days=start_date.weekday())
    # Extend end to Sunday
    end_date = end_date + timedelta(days=(6 - end_date.weekday()))

    total_days = (end_date - start_date).days + 1
    num_weeks = total_days // 7

    # Build grid: 7 rows (Mon=0 to Sun=6) x num_weeks columns
    z = [[0] * num_weeks for _ in range(7)]
    hover_text = [[""] * num_weeks for _ in range(7)]
    x_labels = []

    for w in range(num_weeks):
        week_start = start_date + timedelta(weeks=w)
        x_labels.append(week_start.strftime("%b %d"))
        for d in range(7):
            day = week_start + timedelta(days=d)
            day_str = day.strftime("%Y-%m-%d")
            count = date_counts.get(day_str, 0)
            z[d][w] = count
            hover_text[d][w] = f"{day.strftime('%b %d, %Y')}: {count} commit{'s' if count != 1 else ''}"

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Stats
    most_active = max(date_counts.items(), key=lambda x: x[1])

    # Longest streak
    sorted_dates_dt = sorted([datetime.strptime(d, "%Y-%m-%d") for d in date_counts.keys()])
    longest_streak = 1
    current_streak = 1
    for i in range(1, len(sorted_dates_dt)):
        if (sorted_dates_dt[i] - sorted_dates_dt[i - 1]).days == 1:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 1

    return {
        'grid_z': z,
        'grid_hover': hover_text,
        'x_labels': x_labels,
        'y_labels': day_names,
        'most_active_day': {'date': most_active[0], 'count': most_active[1]},
        'longest_streak': longest_streak,
        'total_active_days': len(date_counts),
    }


def generate_health_summary(drift_data: dict, ghosts: list, hot_modules: list, chunks: list) -> dict:
    """Generate a rule-based plain English health report.

    Returns:
        dict with health_score, verdict, insights, and stats
    """
    if not chunks:
        return None

    total_commits = len(set(c['commit_hash'] for c in chunks))
    total_files = len(set(c['file_name'] for c in chunks))
    total_functions = len(set(f"{c['file_name']}::{c['chunk_name']}" for c in chunks))
    ghost_count = len(ghosts)

    # Compute average drift across all files
    all_drift_scores = []
    for fname, commits in drift_data.items():
        for c in commits:
            all_drift_scores.append(c['drift_score'])

    if all_drift_scores:
        avg_drift = float(np.mean(all_drift_scores))
    else:
        avg_drift = 0.0

    health_score = int((1 - min(avg_drift, 1.0)) * 100)

    if health_score >= 80:
        verdict = "Healthy"
        verdict_color = "#4A5C48"
        verdict_desc = "This codebase shows strong stability. Architecture is well-maintained."
    elif health_score >= 60:
        verdict = "Aging"
        verdict_color = "#AF6384"
        verdict_desc = "Some modules are drifting. Worth reviewing volatile areas."
    else:
        verdict = "Decaying"
        verdict_color = "#F15D90"
        verdict_desc = "High semantic drift detected. Major refactoring may be needed."

    # Date range
    dates = [c['commit_date'][:10] for c in chunks]
    date_range_start = min(dates) if dates else "N/A"
    date_range_end = max(dates) if dates else "N/A"

    most_stable = hot_modules[-1]['file_name'] if hot_modules else 'N/A'
    most_volatile = hot_modules[0]['file_name'] if hot_modules else 'N/A'

    # Build insight sentences
    insights = []
    insights.append(
        f"The codebase has been active across **{total_commits}** commits "
        f"spanning **{date_range_start}** to **{date_range_end}**."
    )
    if most_volatile != 'N/A':
        insights.append(
            f"Semantic drift is highest in **{most_volatile}** — consider reviewing for refactoring."
        )
    if ghost_count > 0:
        insights.append(
            f"**{ghost_count}** ghost concept{'s' if ghost_count != 1 else ''} detected — "
            f"features that were abandoned mid-development."
        )
    else:
        insights.append("No ghost concepts found — your codebase retains its institutional knowledge well.")
    if most_stable != 'N/A':
        insights.append(
            f"**{most_stable}** has remained architecturally consistent throughout the history."
        )
    insights.append(
        f"**{total_functions}** unique functions/classes analyzed across **{total_files}** files."
    )

    return {
        'health_score': health_score,
        'verdict': verdict,
        'verdict_color': verdict_color,
        'verdict_desc': verdict_desc,
        'total_commits': total_commits,
        'total_files': total_files,
        'total_functions': total_functions,
        'ghost_count': ghost_count,
        'most_stable': most_stable,
        'most_volatile': most_volatile,
        'avg_drift': avg_drift,
        'insights': insights,
        'date_range': f"{date_range_start} → {date_range_end}",
    }
