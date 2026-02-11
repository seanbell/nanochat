"""
SFT Curation Browser — browse clusters, conversations, and IFD scores from curated data.

Works at any stage of the curation pipeline:
  - If the final JSONL exists, loads it directly (fastest, full metadata).
  - If JSONL is missing but --dataset is provided, loads the source HF dataset
    with the same filtering as curate_sft.py, then overlays any numpy checkpoints
    (clusters, IFD) that exist so far.
  - If neither JSONL nor --dataset, shows stats-only from numpy arrays.

Launch:
  python -m scripts.curate_sft_web --output-name dolci_curated_500k --dataset allenai/Dolci-Instruct-SFT
  python -m scripts.curate_sft_web --output-name test_curated --port 8001
"""

import argparse
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from nanochat.common import load_curated_data

parser = argparse.ArgumentParser(description="SFT Curation Browser")
parser.add_argument("--output-name", required=True, help="Name of curated output (matches curate_sft --output-name)")
parser.add_argument("--dataset", default=None, help="Source HF dataset (for browsing conversations when JSONL not yet produced)")
parser.add_argument("--downsample", type=int, default=0, help="Must match --downsample used in curate_sft (0 = none)")
parser.add_argument("--seed", type=int, default=42, help="Must match --seed used in curate_sft")
parser.add_argument("--port", type=int, default=8001, help="Port to run the server on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data at startup — whatever exists
# ---------------------------------------------------------------------------
data = load_curated_data(args.output_name, args.dataset, args.downsample, args.seed)
conversations = data["conversations"]
has_jsonl = data["has_jsonl"]
n_total = data["n_total"]
completed_stages = data["completed_stages"]
has_conversations = len(conversations) > 0

# ---------------------------------------------------------------------------
# Build indices and stats from loaded conversations (metadata already injected)
# ---------------------------------------------------------------------------
cluster_to_indices = defaultdict(list)
ifd_values_list = []

for i, conv in enumerate(conversations):
    if "cluster" in conv:
        cluster_to_indices[conv["cluster"]].append(i)
    if "ifd_diff" in conv:
        ifd_values_list.append(conv["ifd_diff"])

# Precompute cluster stats
cluster_stats = []
for cid in sorted(cluster_to_indices.keys()):
    indices = cluster_to_indices[cid]
    ifds = [conversations[i]["ifd_diff"] for i in indices if "ifd_diff" in conversations[i]]
    cluster_stats.append({
        "cluster": cid,
        "count": len(indices),
        "mean_ifd": round(sum(ifds) / len(ifds), 4) if ifds else None,
        "min_ifd": round(min(ifds), 4) if ifds else None,
        "max_ifd": round(max(ifds), 4) if ifds else None,
    })

# Precompute IFD histogram (20 bins)
ifd_histogram = []
if ifd_values_list:
    ifd_min_val, ifd_max_val = min(ifd_values_list), max(ifd_values_list)
    n_bins = 20
    bin_width = (ifd_max_val - ifd_min_val) / n_bins if ifd_max_val > ifd_min_val else 1
    bins = [0] * n_bins
    for v in ifd_values_list:
        b = min(int((v - ifd_min_val) / bin_width), n_bins - 1)
        bins[b] += 1
    for i, count in enumerate(bins):
        ifd_histogram.append({
            "lo": round(ifd_min_val + i * bin_width, 4),
            "hi": round(ifd_min_val + (i + 1) * bin_width, 4),
            "count": count,
        })

print(f"Ready: {len(conversations):,} conversations, {len(cluster_stats)} clusters, "
      f"stages: {', '.join(completed_stages) or 'none'}")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/")
async def root():
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/stats")
async def api_stats():
    n = len(ifd_values_list)
    sorted_ifd = sorted(ifd_values_list) if n > 0 else []
    return {
        "total_source": n_total,  # full filtered dataset size (from numpy arrays)
        "total_shown": len(conversations),  # what we're browsing
        "has_conversations": has_conversations,
        "has_jsonl": has_jsonl,
        "clusters": len(cluster_stats),
        "completed_stages": completed_stages,
        "ifd_histogram": ifd_histogram,
        "ifd_mean": round(sum(ifd_values_list) / n, 4) if n else None,
        "ifd_p10": round(sorted_ifd[n // 10], 4) if n else None,
        "ifd_p90": round(sorted_ifd[n * 9 // 10], 4) if n else None,
    }


@app.get("/api/clusters")
async def api_clusters():
    return cluster_stats


@app.get("/api/conversations")
async def api_conversations(
    page: int = 0,
    per_page: int = 50,
    sort: str = "index",
    cluster: int | None = None,
    q: str | None = None,
    ifd_min: float | None = None,
    ifd_max: float | None = None,
):
    if not has_conversations:
        return {"total": 0, "page": 0, "per_page": per_page, "items": [],
                "error": "No conversations loaded. Pass --dataset to browse source data."}

    # Start with all indices or cluster-filtered
    if cluster is not None:
        indices = list(cluster_to_indices.get(cluster, []))
    else:
        indices = list(range(len(conversations)))

    # Text search filter
    if q:
        q_lower = q.lower()
        indices = [i for i in indices if any(
            q_lower in msg.get("content", "").lower()
            for msg in conversations[i].get("messages", [])
        )]

    # IFD range filter
    if ifd_min is not None:
        indices = [i for i in indices if conversations[i].get("ifd_diff", float("-inf")) >= ifd_min]
    if ifd_max is not None:
        indices = [i for i in indices if conversations[i].get("ifd_diff", float("inf")) <= ifd_max]

    # Sort — parse "field_direction" pattern
    if sort.endswith("_desc"):
        reverse = True
        field = sort[:-5]
    elif sort.endswith("_asc"):
        reverse = False
        field = sort[:-4]
    else:
        reverse, field = False, sort

    if field == "ifd":
        indices.sort(key=lambda i: conversations[i].get("ifd_diff", 0), reverse=reverse)
    elif field == "length":
        indices.sort(key=lambda i: sum(len(m.get("content", "")) for m in conversations[i].get("messages", [])), reverse=reverse)
    elif field == "cluster":
        indices.sort(key=lambda i: conversations[i].get("cluster", 0), reverse=reverse)
    elif field == "index":
        indices.sort(reverse=reverse)

    total = len(indices)
    page_indices = indices[page * per_page : (page + 1) * per_page]

    items = []
    for i in page_indices:
        conv = conversations[i]
        msgs = conv.get("messages", [])
        preview = msgs[0]["content"][:120] if msgs else ""
        n_turns = sum(1 for m in msgs if m["role"] == "assistant")
        items.append({
            "index": i,
            "preview": preview,
            "n_turns": n_turns,
            "n_messages": len(msgs),
            "cluster": conv.get("cluster"),
            "ifd_diff": conv.get("ifd_diff"),
        })

    return {"total": total, "page": page, "per_page": per_page, "items": items}


_tokenizer = None
def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from nanochat.tokenizer import get_tokenizer
        _tokenizer = get_tokenizer()
    return _tokenizer


@app.get("/api/conversations/{idx}")
async def api_conversation(idx: int):
    if not has_conversations or idx < 0 or idx >= len(conversations):
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = conversations[idx]
    tok = _get_tokenizer()
    # Per-message token counts (content only, no special tokens)
    messages = conv.get("messages", [])
    msg_tokens = [len(tok.encode(m.get("content", ""))) for m in messages]
    # Full conversation token count (with all special tokens + BOS)
    ids, mask = tok.render_conversation(conv)
    return {
        "index": idx,
        "cluster": conv.get("cluster"),
        "ifd_diff": conv.get("ifd_diff"),
        "messages": messages,
        "msg_tokens": msg_tokens,
        "total_tokens": len(ids),
    }


# ---------------------------------------------------------------------------
# Inline HTML SPA
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SFT Curation Browser</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: ui-sans-serif, -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
    background: #ffffff; color: #111827;
    min-height: 100vh; display: flex; flex-direction: column;
}

/* Nav */
nav {
    background: #111827; color: #fff; padding: 0.75rem 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    position: sticky; top: 0; z-index: 100;
}
nav .brand { font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em; }
nav a { color: #d1d5db; text-decoration: none; font-size: 0.9rem; padding: 0.25rem 0.5rem; border-radius: 0.375rem; transition: color 0.15s, background 0.15s; }
nav a:hover, nav a.active { color: #fff; background: rgba(255,255,255,0.1); }

/* Layout */
.container { max-width: 72rem; margin: 0 auto; padding: 1.5rem; width: 100%; }

/* Cards */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1.25rem; }
.stat-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
.stat-card .value { font-size: 1.5rem; font-weight: 700; color: #111827; }

/* Histogram */
.histogram { display: flex; align-items: flex-end; gap: 2px; height: 120px; margin-bottom: 0.5rem; }
.histogram .bar { flex: 1; background: #2563eb; border-radius: 2px 2px 0 0; min-width: 4px; transition: background 0.15s; position: relative; }
.histogram .bar:hover { background: #1d4ed8; }
.histogram-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280; }
.chart-title { font-size: 0.85rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem; }

/* Table */
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 2px solid #e5e7eb; color: #6b7280; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { color: #111827; }
td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #f3f4f6; }
tr:hover td { background: #f9fafb; }
.clickable { cursor: pointer; }

/* Badge */
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
.badge-blue { background: #dbeafe; color: #1e40af; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-gray { background: #f3f4f6; color: #374151; }
.badge-amber { background: #fef3c7; color: #92400e; }

/* Pipeline stages */
.stages { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.stage { padding: 0.3rem 0.6rem; border-radius: 0.375rem; font-size: 0.8rem; font-weight: 500; }
.stage.done { background: #d1fae5; color: #065f46; }
.stage.pending { background: #f3f4f6; color: #9ca3af; }

/* Pagination */
.pagination { display: flex; gap: 0.5rem; align-items: center; justify-content: center; margin-top: 1rem; }
.pagination button { padding: 0.4rem 0.8rem; border: 1px solid #d1d5db; border-radius: 0.375rem; background: #fff; cursor: pointer; font-size: 0.85rem; }
.pagination button:hover:not(:disabled) { background: #f3f4f6; }
.pagination button:disabled { opacity: 0.4; cursor: not-allowed; }
.pagination .info { font-size: 0.85rem; color: #6b7280; }

/* Filters bar */
.filters { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem; }
.filters input, .filters select { padding: 0.45rem 0.7rem; border: 1px solid #d1d5db; border-radius: 0.5rem; font-size: 0.85rem; outline: none; }
.filters input:focus, .filters select:focus { border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,0.15); }
.filters input[type="text"] { width: 220px; }

/* Conversation detail */
.conv-header { margin-bottom: 1.5rem; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
.conv-header .back { color: #2563eb; text-decoration: none; font-size: 0.9rem; cursor: pointer; }
.conv-header .back:hover { text-decoration: underline; }

.message-list { display: flex; flex-direction: column; gap: 0.75rem; max-width: 48rem; }
.message { display: flex; margin-bottom: 0.5rem; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }
.message .bubble { line-height: 1.6; max-width: 80%; word-break: break-word; }
.message.user .bubble { background: #f3f4f6; border-radius: 1.25rem; padding: 0.8rem 1rem; }
.message.assistant .bubble { background: transparent; padding: 0.5rem; }
.message .role-tag { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #9ca3af; margin-bottom: 0.15rem; }

/* Tooltip */
.bar-tooltip { position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: #111827; color: #fff; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.7rem; white-space: nowrap; pointer-events: none; opacity: 0; transition: opacity 0.15s; }
.bar:hover .bar-tooltip { opacity: 1; }

/* Warning */
.warning { background: #fef3c7; border: 1px solid #fcd34d; border-radius: 0.5rem; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #92400e; }
</style>
</head>
<body>
<nav>
    <span class="brand">SFT Curation Browser</span>
    <a href="#/" id="nav-dashboard">Dashboard</a>
    <a href="#/clusters" id="nav-clusters">Clusters</a>
    <a href="#/conversations" id="nav-conversations">Conversations</a>
</nav>
<div class="container" id="app"></div>

<script>
const $ = id => document.getElementById(id);
const app = $('app');
let statsCache = null;
let clustersCache = null;

let convState = { page: 0, sort: 'index', cluster: null, q: '', ifd_min: null, ifd_max: null };

async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`API error: ${r.status}`);
    return r.json();
}

function setActiveNav(id) {
    document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
    const el = $(id);
    if (el) el.classList.add('active');
}

// --- Router ---
async function route() {
    const hash = location.hash || '#/';
    if (hash === '#/' || hash === '#') {
        setActiveNav('nav-dashboard');
        await renderDashboard();
    } else if (hash === '#/clusters') {
        setActiveNav('nav-clusters');
        await renderClusters();
    } else if (hash.startsWith('#/conversations')) {
        setActiveNav('nav-conversations');
        await renderConversations();
    } else if (hash.startsWith('#/conversation/')) {
        setActiveNav('nav-conversations');
        const idx = parseInt(hash.split('/')[2]);
        await renderConversation(idx);
    }
}
window.addEventListener('hashchange', route);

// --- Dashboard ---
async function renderDashboard() {
    if (!statsCache) statsCache = await api('/api/stats');
    const s = statsCache;

    let html = '';

    // Pipeline stage indicators
    const allStages = ['embeddings', 'clusters', 'ifd', 'jsonl'];
    html += '<div class="stages">';
    for (const st of allStages) {
        const done = s.completed_stages.includes(st);
        html += `<span class="stage ${done ? 'done' : 'pending'}">${done ? '\u2713' : '\u00b7'} ${st}</span>`;
    }
    html += '</div>';

    if (!s.has_conversations) {
        html += '<div class="warning">No conversations loaded. Pass --dataset to browse source data while curation runs.</div>';
    } else if (s.has_jsonl) {
        const pct = s.total_source > 0 ? (100 * s.total_shown / s.total_source).toFixed(1) : '?';
        html += `<div style="background:#d1fae5;border:1px solid #6ee7b7;border-radius:0.5rem;padding:0.75rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#065f46">Showing <b>${s.total_shown.toLocaleString()} selected</b> conversations out of ${s.total_source.toLocaleString()} source (${pct}% kept). Selection: cluster-balanced, preferring most negative IFD diff.</div>`;
    } else {
        html += '<div class="warning">Showing source dataset (pre-selection). Final curated JSONL not yet produced.</div>';
    }

    html += '<div class="stats-grid">';
    if (s.total_source > 0) html += statCard('Source Dataset', s.total_source.toLocaleString());
    html += statCard(s.has_jsonl ? 'Selected' : 'Conversations', s.total_shown.toLocaleString());
    if (s.has_jsonl && s.total_source > 0) {
        const pct = (100 * s.total_shown / s.total_source).toFixed(1);
        html += statCard('Selection Rate', pct + '%');
    }
    html += statCard('Clusters', s.clusters.toLocaleString());
    if (s.ifd_mean !== null) {
        html += statCard('Mean IFD Diff', s.ifd_mean.toFixed(4));
        html += statCard('IFD p10 / p90', `${s.ifd_p10.toFixed(3)} / ${s.ifd_p90.toFixed(3)}`);
    }
    html += '</div>';

    // IFD histogram
    if (s.ifd_histogram && s.ifd_histogram.length > 0) {
        const maxCount = Math.max(...s.ifd_histogram.map(b => b.count));
        html += '<div class="chart-title">IFD Diff Distribution (more negative = instruction helps more = better)</div>';
        html += '<div class="histogram">';
        for (const bin of s.ifd_histogram) {
            const h = maxCount > 0 ? (bin.count / maxCount * 100) : 0;
            html += `<div class="bar" style="height:${Math.max(h, 1)}%"><span class="bar-tooltip">${bin.lo.toFixed(3)} - ${bin.hi.toFixed(3)}: ${bin.count.toLocaleString()}</span></div>`;
        }
        html += '</div>';
        html += `<div class="histogram-labels"><span>${s.ifd_histogram[0].lo.toFixed(3)}</span><span>${s.ifd_histogram[s.ifd_histogram.length-1].hi.toFixed(3)}</span></div>`;
    }

    // Cluster size distribution
    if (!clustersCache && s.clusters > 0) clustersCache = await api('/api/clusters');
    if (clustersCache && clustersCache.length > 0) {
        const sorted = [...clustersCache].sort((a, b) => b.count - a.count);
        const maxClCount = sorted[0].count;
        html += '<div style="margin-top:2rem"><div class="chart-title">Cluster Size Distribution (sorted by size)</div>';
        html += '<div class="histogram">';
        for (const cl of sorted) {
            const h = maxClCount > 0 ? (cl.count / maxClCount * 100) : 0;
            html += `<div class="bar" style="height:${Math.max(h, 1)}%;background:#10b981"><span class="bar-tooltip">Cluster ${cl.cluster}: ${cl.count} convs, IFD ${cl.mean_ifd !== null ? cl.mean_ifd.toFixed(3) : 'N/A'}</span></div>`;
        }
        html += '</div>';
        html += `<div class="histogram-labels"><span>Largest</span><span>Smallest</span></div></div>`;
    }

    app.innerHTML = html;
}

function statCard(label, value) {
    return `<div class="stat-card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

// --- Clusters ---
async function renderClusters() {
    if (!clustersCache) clustersCache = await api('/api/clusters');
    const clusters = clustersCache;

    if (clusters.length === 0) {
        app.innerHTML = '<div class="warning">No cluster data available yet. Clustering stage has not completed.</div>';
        return;
    }

    // Sort clusters
    const sortKey = clusterSort.field;
    const rev = clusterSort.desc;
    const sorted = [...clusters].sort((a, b) => {
        const va = a[sortKey] ?? 0, vb = b[sortKey] ?? 0;
        return rev ? vb - va : va - vb;
    });

    const cols = [
        ['Cluster', 'cluster'], ['Count', 'count'], ['Mean IFD', 'mean_ifd'],
        ['Min IFD', 'min_ifd'], ['Max IFD', 'max_ifd'],
    ];

    let html = '<h2 style="font-size:1.1rem;font-weight:600;margin-bottom:1rem">Clusters</h2>';
    html += '<table><thead><tr>';
    for (const [label, key] of cols) {
        const active = clusterSort.field === key;
        const arrow = active ? (clusterSort.desc ? ' \u25bc' : ' \u25b2') : '';
        const style = active ? 'color:#111827' : '';
        html += `<th onclick="sortClusters('${key}')" style="${style}">${label}${arrow}</th>`;
    }
    html += '<th></th></tr></thead><tbody>';

    for (const cl of sorted) {
        html += `<tr class="clickable" onclick="browseCluster(${cl.cluster})">`;
        html += `<td><span class="badge badge-blue">${cl.cluster}</span></td>`;
        html += `<td>${cl.count.toLocaleString()}</td>`;
        html += `<td>${cl.mean_ifd !== null ? cl.mean_ifd.toFixed(4) : '-'}</td>`;
        html += `<td>${cl.min_ifd !== null ? cl.min_ifd.toFixed(4) : '-'}</td>`;
        html += `<td>${cl.max_ifd !== null ? cl.max_ifd.toFixed(4) : '-'}</td>`;
        html += `<td style="color:#2563eb;font-size:0.8rem">Browse &rarr;</td>`;
        html += '</tr>';
    }
    html += '</tbody></table>';
    app.innerHTML = html;
}

let clusterSort = { field: 'cluster', desc: false };
function sortClusters(key) {
    if (clusterSort.field === key) clusterSort.desc = !clusterSort.desc;
    else { clusterSort.field = key; clusterSort.desc = false; }
    renderClusters();
}

function browseCluster(cid) {
    convState.cluster = cid;
    convState.page = 0;
    location.hash = '#/conversations';
}

// --- Conversation list ---
async function renderConversations() {
    const hashParts = location.hash.split('?');
    if (hashParts.length > 1) {
        const params = new URLSearchParams(hashParts[1]);
        if (params.has('cluster')) convState.cluster = parseInt(params.get('cluster'));
    }

    let qp = `page=${convState.page}&per_page=50&sort=${convState.sort}`;
    if (convState.cluster !== null) qp += `&cluster=${convState.cluster}`;
    if (convState.q) qp += `&q=${encodeURIComponent(convState.q)}`;
    if (convState.ifd_min !== null) qp += `&ifd_min=${convState.ifd_min}`;
    if (convState.ifd_max !== null) qp += `&ifd_max=${convState.ifd_max}`;

    const data = await api(`/api/conversations?${qp}`);

    let html = '<h2 style="font-size:1.1rem;font-weight:600;margin-bottom:1rem">Conversations</h2>';

    if (data.error) {
        html += `<div class="warning">${escHtml(data.error)}</div>`;
        app.innerHTML = html;
        return;
    }

    // Filters
    html += '<div class="filters">';
    html += `<input type="text" id="f-search" placeholder="Search messages..." value="${escHtml(convState.q || '')}">`;
    if (convState.cluster !== null) {
        html += `<span class="badge badge-blue">Cluster ${convState.cluster}</span>`;
        html += `<a href="javascript:void(0)" onclick="clearCluster()" style="font-size:0.8rem;color:#2563eb">Clear</a>`;
    }
    html += `<span class="info" style="margin-left:auto">${data.total.toLocaleString()} results</span>`;
    html += '</div>';

    // Table with sortable headers
    const sortCols = {'#':'index', 'Preview':null, 'Turns':'length_desc', 'Cluster':'cluster_asc', 'IFD Diff':'ifd_asc'};
    html += '<table><thead><tr>';
    for (const [label, sortKey] of Object.entries(sortCols)) {
        if (!sortKey) { html += `<th>${label}</th>`; continue; }
        // Toggle direction if clicking the same column
        const isAsc = convState.sort === sortKey;
        const isDesc = convState.sort === sortKey.replace('_asc','_desc');
        const nextSort = isAsc ? sortKey.replace('_asc','_desc') : sortKey;
        const arrow = isAsc ? ' \u25b2' : isDesc ? ' \u25bc' : '';
        const activeStyle = (isAsc || isDesc) ? 'color:#111827' : '';
        html += `<th onclick="sortConv('${nextSort}')" style="${activeStyle}">${label}${arrow}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const item of data.items) {
        html += `<tr class="clickable" onclick="location.hash='#/conversation/${item.index}'">`;
        html += `<td style="color:#6b7280;font-size:0.85rem">${item.index}</td>`;
        html += `<td style="max-width:28rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(item.preview)}</td>`;
        html += `<td>${item.n_turns}</td>`;
        html += `<td>${item.cluster !== null && item.cluster !== undefined ? `<span class="badge badge-blue">${item.cluster}</span>` : '-'}</td>`;
        html += `<td>${item.ifd_diff !== null && item.ifd_diff !== undefined ? item.ifd_diff.toFixed(4) : '-'}</td>`;
        html += '</tr>';
    }
    if (data.items.length === 0) {
        html += '<tr><td colspan="5" style="text-align:center;color:#9ca3af;padding:2rem">No conversations found</td></tr>';
    }
    html += '</tbody></table>';

    // Pagination
    const totalPages = Math.ceil(data.total / 50);
    html += '<div class="pagination">';
    html += `<button onclick="convPage(0)" ${convState.page===0?'disabled':''}>First</button>`;
    html += `<button onclick="convPage(${convState.page - 1})" ${convState.page===0?'disabled':''}>Prev</button>`;
    html += `<span class="info">Page ${convState.page + 1} / ${Math.max(totalPages, 1)}</span>`;
    html += `<button onclick="convPage(${convState.page + 1})" ${convState.page >= totalPages - 1?'disabled':''}>Next</button>`;
    html += `<button onclick="convPage(${totalPages - 1})" ${convState.page >= totalPages - 1?'disabled':''}>Last</button>`;
    html += '</div>';

    app.innerHTML = html;

    // Bind filter events
    const searchEl = $('f-search');
    let searchTimer = null;
    searchEl.addEventListener('input', () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => {
            convState.q = searchEl.value;
            convState.page = 0;
            renderConversations();
        }, 300);
    });
}

function sortConv(key) { convState.sort = key; convState.page = 0; renderConversations(); }

function convPage(p) { convState.page = Math.max(0, p); renderConversations(); }
function clearCluster() { convState.cluster = null; convState.page = 0; renderConversations(); }

// --- Conversation detail ---
async function renderConversation(idx) {
    const data = await api(`/api/conversations/${idx}`);

    let html = '<div class="conv-header">';
    html += `<a class="back" onclick="history.back()">&larr; Back</a>`;
    html += `<span style="font-weight:600">Conversation #${data.index}</span>`;
    if (data.cluster !== null && data.cluster !== undefined) {
        html += `<span class="badge badge-blue">Cluster ${data.cluster}</span>`;
    }
    if (data.ifd_diff !== null && data.ifd_diff !== undefined) {
        html += `<span class="badge badge-green">IFD ${data.ifd_diff.toFixed(4)}</span>`;
    }
    html += `<span class="badge badge-gray">${data.messages.length} messages</span>`;
    if (data.total_tokens) {
        html += `<span class="badge badge-amber">${data.total_tokens.toLocaleString()} tokens (with special)</span>`;
    }
    html += '</div>';

    if (data.total_tokens && data.msg_tokens) {
        const contentSum = data.msg_tokens.reduce((a, b) => a + b, 0);
        const overhead = data.total_tokens - contentSum;
        html += `<div style="font-size:0.8rem;color:#6b7280;margin-bottom:1rem">Per-message counts are content tokens only. Total includes ${overhead} special tokens (BOS + role delimiters).</div>`;
    }

    html += '<div class="message-list">';
    for (let mi = 0; mi < data.messages.length; mi++) {
        const msg = data.messages[mi];
        const role = msg.role || 'unknown';
        const toks = data.msg_tokens ? data.msg_tokens[mi] : null;
        html += `<div class="message ${role}">`;
        html += `<div><div class="role-tag">${role}${toks !== null ? ` \u00b7 ${toks.toLocaleString()} tokens` : ''}</div>`;
        html += `<div class="bubble">${escHtml(msg.content || '').replace(/\n/g, '<br>')}</div>`;
        html += '</div></div>';
    }
    html += '</div>';

    app.innerHTML = html;
}

function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Boot
route();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print(f"Starting SFT Curation Browser on http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
