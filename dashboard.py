# Neuromancer Dashboard — EvoLLM Research Monitor
# Run: streamlit run dashboard.py

import streamlit as st
import json
import glob
import subprocess
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEUROMANCER // EvoLLM",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --bg:        #020408;
    --surface:   #060d12;
    --border:    #0d2a1a;
    --green:     #00ff88;
    --green-dim: #00994d;
    --amber:     #ffb300;
    --red:       #ff3c3c;
    --blue:      #00d4ff;
    --text:      #b0c8b8;
    --dim:       #3a5a46;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Masthead */
.masthead {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 4px;
    color: var(--green);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 4px;
}
.masthead-title {
    font-size: 28px;
    font-weight: 900;
    letter-spacing: 8px;
    color: var(--green);
    text-shadow: 0 0 30px rgba(0,255,136,0.4);
    line-height: 1;
}
.masthead-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: var(--dim);
    letter-spacing: 3px;
    margin-top: 4px;
}

/* Metric tiles */
.metric-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--green);
    padding: 14px 18px;
    font-family: 'Share Tech Mono', monospace;
    position: relative;
    overflow: hidden;
}
.metric-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--green), transparent);
    opacity: 0.6;
}
.metric-label {
    font-size: 9px;
    letter-spacing: 3px;
    color: var(--dim);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--green);
    line-height: 1;
}
.metric-unit {
    font-size: 10px;
    color: var(--dim);
    margin-top: 2px;
}

/* GPU bar */
.gpu-bar-bg {
    background: #0a1a10;
    border: 1px solid var(--border);
    height: 8px;
    border-radius: 2px;
    overflow: hidden;
    margin: 6px 0;
}
.gpu-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
}

/* Section headers */
.section-head {
    font-family: 'Orbitron', monospace;
    font-size: 9px;
    letter-spacing: 4px;
    color: var(--green-dim);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 18px 0 12px;
}

/* Table */
.data-table {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    width: 100%;
    border-collapse: collapse;
}
.data-table th {
    font-size: 9px;
    letter-spacing: 2px;
    color: var(--dim);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
}
.data-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #0a1a0d;
    color: var(--text);
}
.data-table tr:hover td { background: #0a1a10; }

/* Status badges */
.badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.badge-green  { background: #00ff8820; color: var(--green);  border: 1px solid var(--green-dim); }
.badge-amber  { background: #ffb30020; color: var(--amber);  border: 1px solid #996900; }
.badge-red    { background: #ff3c3c20; color: var(--red);    border: 1px solid #991a1a; }
.badge-blue   { background: #00d4ff20; color: var(--blue);   border: 1px solid #006680; }

/* Scanline effect */
.scanlines {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Pulse animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.pulse { animation: pulse 2s infinite; }

[data-testid="stMetricValue"] { color: var(--green) !important; }
[data-testid="column"] { gap: 8px !important; }
div.stButton > button {
    background: transparent;
    border: 1px solid var(--green-dim);
    color: var(--green);
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    border-radius: 2px;
}
div.stButton > button:hover {
    background: #00ff8810;
    border-color: var(--green);
}
</style>
<div class="scanlines"></div>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_gpu_stats():
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        parts = [x.strip() for x in result.stdout.strip().split(',')]
        return {
            'util':    float(parts[0]),
            'mem_used': float(parts[1]),
            'mem_total': float(parts[2]),
            'temp':    float(parts[3]),
            'power':   float(parts[4]) if parts[4] != '[N/A]' else None,
        }
    except Exception:
        return None


def load_experiment_runs(base_dir='experiments'):
    runs = {}
    for run_dir in sorted(glob.glob(f'{base_dir}/phase0_metrics_*/'), reverse=True):
        run_id = Path(run_dir).name.replace('phase0_metrics_', '')
        json_file = os.path.join(run_dir, 'metrics_phase0.json')
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    runs[run_id] = json.load(f)
            except Exception:
                pass
    return runs


def render_bar(value, max_val=100, color_var='--green'):
    pct = min(100, max(0, value / max_val * 100))
    if pct > 85:
        color = 'var(--red)'
    elif pct > 60:
        color = 'var(--amber)'
    else:
        color = f'var({color_var})'
    return f"""
    <div class="gpu-bar-bg">
        <div class="gpu-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
    </div>
    """


def badge(text, kind='green'):
    return f'<span class="badge badge-{kind}">{text}</span>'


# ── Layout ─────────────────────────────────────────────────────────────────────

# Masthead
st.markdown("""
<div class="masthead">Evolutionary LLM Research // Neuromancer Pipeline</div>
<div class="masthead-title">NEUROMANCER</div>
<div class="masthead-sub">adaptive information ecology monitor · qwen3-8b-base · unsloth</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Auto-refresh control
col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
with col_r1:
    auto_refresh = st.checkbox("AUTO REFRESH", value=False)
with col_r2:
    refresh_interval = st.selectbox("INTERVAL", [5, 10, 30, 60], index=1,
                                     label_visibility="collapsed")
with col_r3:
    if st.button("⟳ REFRESH NOW"):
        st.rerun()

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Timestamp
now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
st.markdown(f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:10px;
     color:var(--dim);letter-spacing:2px;margin-bottom:16px;">
     SYSTEM TIME · {now}
</div>
""", unsafe_allow_html=True)


# ── GPU Panel ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// GPU STATUS · RTX 4090</div>',
            unsafe_allow_html=True)

gpu = get_gpu_stats()

if gpu:
    g1, g2, g3, g4, g5 = st.columns(5)

    def tile(col, label, value, unit=''):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    tile(g1, "GPU UTIL",   f"{gpu['util']:.0f}", "%")
    tile(g2, "VRAM USED",  f"{gpu['mem_used']/1024:.1f}", "GB")
    tile(g3, "VRAM FREE",  f"{(gpu['mem_total']-gpu['mem_used'])/1024:.1f}", "GB")
    tile(g4, "TEMP",       f"{gpu['temp']:.0f}", "°C")
    tile(g5, "POWER",      f"{gpu['power']:.0f}" if gpu['power'] else "N/A", "W")

    st.markdown(
        f"**GPU LOAD**{render_bar(gpu['util'])}",
        unsafe_allow_html=True
    )
    st.markdown(
        f"**VRAM**{render_bar(gpu['mem_used'], gpu['mem_total'], '--blue')}",
        unsafe_allow_html=True
    )
    st.markdown(
        f"**TEMPERATURE**{render_bar(gpu['temp'], 95, '--amber')}",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:12px;'
        f'color:var(--dim);">'
        f'{badge("nvidia-smi unavailable", "amber")} — running on Windows host or GPU not detected</div>',
        unsafe_allow_html=True
    )


# ── Phase Status ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// RESEARCH PHASES</div>',
            unsafe_allow_html=True)

phases = [
    ("PRE-0",  "Environment setup",           "complete"),
    ("0",      "Metric validation + calibration", "complete"),
    ("1",      "Information ecology (single model)", "ready"),
    ("2",      "Evolutionary dynamics (population)", "pending"),
    ("3",      "HGT + cannibalism recombination",   "pending"),
    ("4",      "Emergent functional archetypes",     "pending"),
]

ph_cols = st.columns(len(phases))
for col, (phase, desc, status) in zip(ph_cols, phases):
    kind = {'complete': 'green', 'ready': 'blue', 'pending': 'amber'}[status]
    icon = {'complete': '✓', 'ready': '▶', 'pending': '○'}[status]
    with col:
        st.markdown(f"""
        <div class="metric-tile" style="border-top-color:{'var(--green)' if status=='complete' else 'var(--blue)' if status=='ready' else 'var(--dim)'};min-height:90px;">
            <div class="metric-label">PHASE {phase}</div>
            <div style="font-size:22px;color:{'var(--green)' if status=='complete' else 'var(--blue)' if status=='ready' else 'var(--dim)'};font-family:'Share Tech Mono',monospace;">{icon}</div>
            <div style="font-size:10px;color:var(--dim);margin-top:4px;font-family:'Share Tech Mono',monospace;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Experiment Results ─────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// PHASE 0 · METRIC VALIDATION RESULTS</div>',
            unsafe_allow_html=True)

runs = load_experiment_runs()

if not runs:
    # Show demo data from known Phase 0 results
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--dim);">
    No experiments/phase0_metrics_* directories found. Showing canonical Phase 0 results.
    </div>
    """, unsafe_allow_html=True)

    demo_data = {
        'food':     {'H_X': 5.49, 'C_X': 0.529, 'I_X_seed': 0.0873, 'H_dezorg': 0.823, 'fitness': -0.207, 'n': 73},
        'predator': {'H_X': 5.06, 'C_X': 0.408, 'I_X_seed': 0.0445, 'H_dezorg': 0.904, 'fitness': -0.451, 'n': 196},
        'noise':    {'H_X': 5.79, 'C_X': 0.549, 'I_X_seed': 0.0931, 'H_dezorg': 0.905, 'fitness': -0.263, 'n': 35},
    }

    kw_results = {
        'H(X)':      {'H': 36.30, 'p': 1.31e-8},
        'C(X)':      {'H': 80.83, 'p': 2.81e-18},
        'I(X;seed)': {'H': 50.37, 'p': 1.15e-11},
    }

    # Summary table
    rows = []
    for dtype, m in demo_data.items():
        rows.append({
            'TYPE': dtype.upper(),
            'N': m['n'],
            'H(X)': f"{m['H_X']:.3f}",
            'C(X)': f"{m['C_X']:.3f}",
            'I(X;seed)': f"{m['I_X_seed']:.4f}",
            'H_dezorg': f"{m['H_dezorg']:.3f}",
            'FITNESS': f"{m['fitness']:.3f}",
        })

    df = pd.DataFrame(rows)
    st.markdown(df.to_html(index=False, classes='data-table'), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KW results
    kw_cols = st.columns(3)
    for col, (metric, res) in zip(kw_cols, kw_results.items()):
        with col:
            p_str = f"{res['p']:.2e}"
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-label">K-W · {metric}</div>
                <div class="metric-value" style="font-size:18px;">p = {p_str}</div>
                <div style="margin-top:6px;">{badge('SIGNIFICANT p<0.05', 'green')}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                     color:var(--dim);margin-top:4px;">H = {res['H']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    # Fitness weights
    st.markdown('<div class="section-head">// FITNESS FUNCTION · CALIBRATED WEIGHTS</div>',
                unsafe_allow_html=True)

    fw1, fw2, fw3, fw4 = st.columns(4)
    with fw1:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-label">w1 · COMPLEXITY</div>
            <div class="metric-value">0.3</div>
            <div class="metric-unit">C(X) weight</div>
        </div>""", unsafe_allow_html=True)
    with fw2:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-label">w2 · MUTUAL INFO</div>
            <div class="metric-value">0.5</div>
            <div class="metric-unit">I(X;seed) weight</div>
        </div>""", unsafe_allow_html=True)
    with fw3:
        st.markdown("""
        <div class="metric-tile">
            <div class="metric-label">w3 · DISORGANIZATION</div>
            <div class="metric-value">0.2</div>
            <div class="metric-unit">H_dezorg weight</div>
        </div>""", unsafe_allow_html=True)
    with fw4:
        st.markdown("""
        <div class="metric-tile" style="border-top-color:var(--blue);">
            <div class="metric-label">VALIDATION</div>
            <div class="metric-value" style="font-size:16px;color:var(--blue);">LOCKED</div>
            <div class="metric-unit">grid search · sum=1</div>
        </div>""", unsafe_allow_html=True)

else:
    # Real data from JSON files
    run_id = st.selectbox("EXPERIMENT RUN", list(runs.keys()),
                           label_visibility="collapsed")
    data = runs[run_id]
    st.json(data)


# ── Effect Sizes ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// EFFECT SIZES · RANK-BISERIAL r</div>',
            unsafe_allow_html=True)

effect_data = {
    'COMPARISON':      ['food/predator', 'food/noise', 'predator/noise'],
    'H(X)':            ['-0.38', '+0.08', '+0.48'],
    'C(X)':            ['-0.60', '0.00',  '+0.66'],
    'I(X;seed)':       ['-0.45', '+0.07', '+0.57'],
}
ef_df = pd.DataFrame(effect_data)
st.markdown(ef_df.to_html(index=False, classes='data-table'), unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);
     margin-top:8px;line-height:1.6;">
│ |r| &gt; 0.4 = large effect &nbsp;│&nbsp; food/predator C(X) = -0.60 → large separation &nbsp;│
│ food/noise ≈ 0 → noise from food sentences indistinguishable from food (expected) &nbsp;│
</div>
""", unsafe_allow_html=True)


# ── Corpus Summary ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// CORPUS · PHASE 0 DATA</div>',
            unsafe_allow_html=True)

corpus_data = [
    ('food_climate.jsonl',    26,  'PMC peer-reviewed',        'food',     'climate',   'green'),
    ('food_vaccines.jsonl',   23,  'PMC peer-reviewed',        'food',     'vaccines',  'green'),
    ('food_covid.jsonl',      52,  'PMC peer-reviewed',        'food',     'covid-19',  'green'),
    ('predator_climate.jsonl',32,  'ClimateFever REFUTES',     'predator', 'climate',   'amber'),
    ('predator_vaccines.jsonl',23, 'VaccineLies MisT',         'predator', 'vaccines',  'amber'),
    ('predator_covid.jsonl',  141, 'CoAID fake news',          'predator', 'covid-19',  'amber'),
    ('noise.jsonl',           35,  'Shuffled food sentences',  'noise',    'all',       'blue'),
]

rows_html = ''
for fname, n, source, dtype, domain, color in corpus_data:
    rows_html += f"""
    <tr>
        <td style="font-family:'Share Tech Mono',monospace;color:var(--text);">{fname}</td>
        <td style="color:var(--green);">{n}</td>
        <td>{badge(dtype, color)}</td>
        <td style="color:var(--dim);">{domain}</td>
        <td style="font-size:11px;color:var(--dim);">{source}</td>
    </tr>
    """

st.markdown(f"""
<table class="data-table">
<thead><tr>
  <th>FILE</th><th>N</th><th>TYPE</th><th>DOMAIN</th><th>SOURCE</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)


# ── Config snapshot ────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// ACTIVE CONFIGURATION</div>',
            unsafe_allow_html=True)

cfg1, cfg2 = st.columns(2)
with cfg1:
    st.markdown("""
    <div class="metric-tile">
        <div class="metric-label">MODEL</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--text);line-height:1.8;">
        base: unsloth/qwen3-8b-base-unsloth-bnb-4bit<br>
        backend: unsloth · 4-bit quantized<br>
        temperature: 0.0 (greedy · deterministic)<br>
        max_new_tokens: 200<br>
        seed: 42
        </div>
    </div>
    """, unsafe_allow_html=True)
with cfg2:
    st.markdown("""
    <div class="metric-tile">
        <div class="metric-label">PIPELINE</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--text);line-height:1.8;">
        fitness = 0.3·C(X) + 0.5·I(X;seed) − 0.2·H_dezorg<br>
        jaccard: enabled (diagnostic)<br>
        reproduction: disabled<br>
        HGT: disabled<br>
        cannibalism: disabled
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--dim);
     letter-spacing:2px;border-top:1px solid var(--border);padding-top:12px;margin-top:24px;
     text-align:center;">
NEUROMANCER · EVOLUTIONARY LLM RESEARCH · UNIVERSITY OF SILESIA · KATOWICE ·
qwen3:8b-base + unsloth · RTX 4090 24GB · AMD THREADRIPPER 7960X · 256GB RAM
</div>
""", unsafe_allow_html=True)
