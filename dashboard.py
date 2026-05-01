# Neuromancer Dashboard — EvoLLM Research Monitor
# Run: streamlit run dashboard.py
# Requires: pip install streamlit plotly pandas streamlit-autorefresh

import streamlit as st
import json, glob, subprocess, os, time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(
    page_title="NEUROMANCER // EvoLLM",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&family=Orbitron:wght@400;700;900&display=swap');
:root {
    --bg:#020408; --surface:#060d12; --border:#0d2a1a;
    --green:#00ff88; --green-dim:#00994d; --amber:#ffb300;
    --red:#ff3c3c; --blue:#00d4ff; --text:#b0c8b8; --dim:#3a5a46;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'Rajdhani',sans-serif;font-size:17px;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
.masthead{font-family:'Orbitron',monospace;font-size:11px;font-weight:700;letter-spacing:4px;color:var(--green);text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:8px;margin-bottom:4px;}
.masthead-title{font-size:28px;font-weight:900;letter-spacing:8px;color:var(--green);text-shadow:0 0 30px rgba(0,255,136,0.4);line-height:1;}
.masthead-sub{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);letter-spacing:3px;margin-top:4px;}
.metric-tile{background:var(--surface);border:1px solid var(--border);border-top:2px solid var(--green);padding:14px 18px;font-family:'Share Tech Mono',monospace;position:relative;overflow:hidden;}
.metric-tile::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,var(--green),transparent);opacity:0.6;}
.metric-label{font-size:14px;letter-spacing:3px;color:var(--dim);text-transform:uppercase;margin-bottom:4px;}
.metric-value{font-size:14px;font-weight:700;color:var(--green);line-height:1;}
.metric-unit{font-size:14px;color:var(--dim);margin-top:2px;}
.gpu-bar-bg{background:#0a1a10;border:1px solid var(--border);height:8px;border-radius:2px;overflow:hidden;margin:6px 0;}
.gpu-bar-fill{height:100%;border-radius:2px;transition:width 0.3s;}
.section-head{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:4px;color:var(--green-dim);text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:6px;margin:18px 0 12px;}
.badge{display:inline-block;font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:2px;padding:2px 8px;border-radius:2px;text-transform:uppercase;}
.badge-green{background:#00ff8820;color:var(--green);border:1px solid var(--green-dim);}
.badge-amber{background:#ffb30020;color:var(--amber);border:1px solid #996900;}
.badge-red{background:#ff3c3c20;color:var(--red);border:1px solid #991a1a;}
.badge-blue{background:#00d4ff20;color:var(--blue);border:1px solid #006680;}
.scanlines{position:fixed;top:0;left:0;right:0;bottom:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.08) 2px,rgba(0,0,0,0.08) 4px);pointer-events:none;z-index:9999;}
div.stButton>button{background:transparent;border:1px solid var(--green-dim);color:var(--green);font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:2px;border-radius:2px;}
div.stButton>button:hover{background:#00ff8810;border-color:var(--green);}
</style>
<div class="scanlines"></div>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor='#020408', plot_bgcolor='#060d12',
    font=dict(family='Share Tech Mono', color='#b0c8b8', size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)
TYPE_COLORS = {'food': '#00ff88', 'predator': '#ffb300', 'noise': '#00d4ff'}


def get_gpu_stats():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        p = [x.strip() for x in r.stdout.strip().split(',')]
        return {'util': float(p[0]), 'mem_used': float(p[1]), 'mem_total': float(p[2]),
                'temp': float(p[3]), 'power': float(p[4]) if p[4] not in ('[N/A]', 'N/A') else None}
    except Exception:
        return None


def render_bar(value, max_val=100, color_var='--green'):
    pct = min(100, max(0, value / max_val * 100))
    color = 'var(--red)' if pct > 85 else 'var(--amber)' if pct > 60 else f'var({color_var})'
    return f'<div class="gpu-bar-bg"><div class="gpu-bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'


def tile_html(label, value, unit='', color='var(--green)'):
    return f"""<div class="metric-tile" style="border-top-color:{color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-unit">{unit}</div></div>"""


def load_experiments(base_dir='experiments'):
    runs = {}
    for run_dir in sorted(glob.glob(f'{base_dir}/phase0_metrics_*/'), reverse=True):
        run_id = Path(run_dir).name.replace('phase0_metrics_', '')
        jf = os.path.join(run_dir, 'metrics_phase0.json')
        if os.path.exists(jf):
            try:
                with open(jf) as f:
                    data = json.load(f)
                prog = os.path.join(run_dir, 'metrics_progressive.jsonl')
                data['_prog'] = prog if os.path.exists(prog) else None
                run_name = data.get('run', {}).get('name', run_id)
                runs[run_id] = {"data": data, "run_name": run_name}
            except Exception:
                pass
    return runs


def get_mean_metrics(data):
    mm = data.get('mean_metrics', {})
    result = {}
    for dtype, m in mm.items():
        result[dtype] = {k: m.get(k, 0) for k in ['h_x', 'c_x', 'i_x_seed', 'h_dezorg', 'fitness', 'jaccard', 'count']}
    return result


def get_kw(data):
    kw = data.get('kruskal_wallis', {})
    result = {}
    for metric, vals in kw.items():
        result[metric] = {'stat': vals.get('stat', vals.get('statistic', 0)),
                          'p':    vals.get('p', vals.get('pvalue', 1))}
    return result


def get_effects(data):
    return data.get('effect_sizes', {})


def load_progressive(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    except Exception:
        return []


# ── Canonical demo fallback ────────────────────────────────────────────────────
DEMO_METRICS = {
    'food':     {'h_x': 5.503, 'c_x': 0.526, 'i_x_seed': 0.0900, 'h_dezorg': 0.840, 'fitness': 0.035,  'jaccard': 0.018, 'count': 73},
    'predator': {'h_x': 5.240, 'c_x': 0.425, 'i_x_seed': 0.0717, 'h_dezorg': 0.925, 'fitness': -0.024, 'jaccard': 0.021, 'count': 116},
    'noise':    {'h_x': 5.771, 'c_x': 0.564, 'i_x_seed': 0.0912, 'h_dezorg': 0.921, 'fitness': 0.035,  'jaccard': 0.020, 'count': 35},
}
DEMO_KW = {
    'h_x':      {'stat': 23.55, 'p': 7.68e-6},
    'c_x':      {'stat': 52.99, 'p': 3.12e-12},
    'i_x_seed': {'stat': 5.59,  'p': 0.061},
    'jaccard':  {'stat': 2.26,  'p': 0.323},
}
DEMO_EFFECTS = {
    'h_x':      {'food_vs_predator': -0.357, 'food_vs_noise': 0.033, 'predator_vs_noise': 0.410},
    'c_x':      {'food_vs_predator': -0.524, 'food_vs_noise': 0.107, 'predator_vs_noise': 0.633},
    'i_x_seed': {'food_vs_predator': -0.168, 'food_vs_noise': 0.033, 'predator_vs_noise': 0.209},
}
DEMO_CORPUS = [
    ('food_climate.jsonl',     26, 'PMC peer-reviewed',    'food',     'climate'),
    ('food_vaccines.jsonl',    23, 'PMC peer-reviewed',    'food',     'vaccines'),
    ('food_covid.jsonl',       52, 'PMC peer-reviewed',    'food',     'covid-19'),
    ('predator_climate.jsonl', 32, 'ClimateFever REFUTES', 'predator', 'climate'),
    ('predator_vaccines.jsonl',23, 'VaccineLies MisT',     'predator', 'vaccines'),
    ('predator_covid.jsonl',   61, 'CoAID + synthetic',    'predator', 'covid-19'),
    ('noise.jsonl',            35, '50/50 food+predator',  'noise',    'all'),
]


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="masthead">Evolutionary LLM Research // Neuromancer Pipeline</div>
<div class="masthead-title">NEUROMANCER</div>
<div class="masthead-sub">adaptive information ecology monitor · qwen3-8b-base · unsloth</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Auto-refresh controls ──────────────────────────────────────────────────────
cr1, cr2, cr3 = st.columns([2, 1, 1])
with cr1:
    auto_refresh = st.checkbox("AUTO REFRESH", value=False)
with cr2:
    refresh_interval = st.selectbox("INTERVAL (s)", [5, 10, 30, 60], index=1, label_visibility="collapsed")
with cr3:
    if st.button("⟳ REFRESH NOW"):
        st.rerun()

# Use streamlit-autorefresh if available (no screen flash)
if auto_refresh:
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")
    else:
        st.warning("Install streamlit-autorefresh for flicker-free refresh: pip install streamlit-autorefresh")
        time.sleep(refresh_interval)
        st.rerun()

now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:var(--dim);letter-spacing:2px;margin-bottom:16px;">SYSTEM TIME · {now}</div>', unsafe_allow_html=True)


# ── GPU ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// GPU STATUS · RTX 4090</div>', unsafe_allow_html=True)
gpu = get_gpu_stats()
if gpu:
    g1, g2, g3, g4, g5 = st.columns(5)
    temp_color = 'var(--red)' if gpu['temp'] > 80 else 'var(--amber)' if gpu['temp'] > 65 else 'var(--green)'
    g1.markdown(tile_html("GPU UTIL",  f"{gpu['util']:.0f}", "%"), unsafe_allow_html=True)
    g2.markdown(tile_html("VRAM USED", f"{gpu['mem_used']/1024:.1f}", "GB", 'var(--blue)'), unsafe_allow_html=True)
    g3.markdown(tile_html("VRAM FREE", f"{(gpu['mem_total']-gpu['mem_used'])/1024:.1f}", "GB"), unsafe_allow_html=True)
    g4.markdown(tile_html("TEMP",      f"{gpu['temp']:.0f}", "°C", temp_color), unsafe_allow_html=True)
    g5.markdown(tile_html("POWER",     f"{gpu['power']:.0f}" if gpu['power'] else "N/A", "W"), unsafe_allow_html=True)
    st.markdown(f"**GPU LOAD**{render_bar(gpu['util'])}", unsafe_allow_html=True)
    st.markdown(f"**VRAM**{render_bar(gpu['mem_used'], gpu['mem_total'], '--blue')}", unsafe_allow_html=True)
    st.markdown(f"**TEMPERATURE**{render_bar(gpu['temp'], 95, '--amber')}", unsafe_allow_html=True)
else:
    st.warning("nvidia-smi unavailable")


# ── Phases ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// RESEARCH PHASES</div>', unsafe_allow_html=True)
phases = [
    ("PRE-0", "Environment setup",                  "complete"),
    ("0",     "Metric validation + calibration",    "complete"),
    ("1",     "Information ecology (single model)", "ready"),
    ("2",     "Evolutionary dynamics (population)", "pending"),
    ("3",     "HGT + cannibalism recombination",    "pending"),
    ("4",     "Emergent functional archetypes",      "pending"),
]
ph_cols = st.columns(len(phases))
for col, (phase, desc, status) in zip(ph_cols, phases):
    border = {'complete': 'var(--green)', 'ready': 'var(--blue)', 'pending': 'var(--dim)'}[status]
    icon   = {'complete': '✓', 'ready': '▶', 'pending': '○'}[status]
    col.markdown(f"""<div class="metric-tile" style="border-top-color:{border};min-height:90px;">
        <div class="metric-label">PHASE {phase}</div>
        <div style="font-size:22px;color:{border};font-family:'Share Tech Mono',monospace;">{icon}</div>
        <div style="font-size:10px;color:var(--dim);margin-top:4px;font-family:'Share Tech Mono',monospace;">{desc}</div>
    </div>""", unsafe_allow_html=True)


# ── Progressive logging ────────────────────────────────────────────────────────
prog_files = sorted(glob.glob('experiments/*/metrics_progressive.jsonl'))
if prog_files:
    prog_lines = load_progressive(prog_files[-1])
    run_dir = os.path.dirname(prog_files[-1])
    jf = os.path.join(run_dir, 'metrics_phase0.json')
    total_docs = None
    if os.path.exists(jf):
        try:
            with open(jf) as f:
                pdata = json.load(f)
            dc = pdata.get('run', {}).get('doc_count', {})
            if dc:
                total_docs = sum(dc.values())
        except Exception:
            pass

    st.markdown('<div class="section-head">// LIVE PROGRESS</div>', unsafe_allow_html=True)
    processed = len(prog_lines)
    if total_docs:
        pct = processed / total_docs
        pc1, pc2, pc3 = st.columns([3, 1, 1])
        with pc1:
            st.progress(pct, text=f"Processing: {processed}/{total_docs} documents ({pct*100:.1f}%)")
        pc2.markdown(tile_html("PROCESSED", str(processed), "", 'var(--blue)'), unsafe_allow_html=True)
        pc3.markdown(tile_html("REMAINING", str(total_docs - processed)), unsafe_allow_html=True)
    else:
        st.info(f"Progressive log: {processed} documents processed")

    if prog_lines:
        last = prog_lines[-1]
        st.markdown(f"""<div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--dim);margin-top:4px;">
        LAST: {last.get('id', last.get('doc_id','?'))} | type={last.get('type','?')} |
        H={last.get('h_x',0):.3f} | C={last.get('c_x',0):.3f} |
        I={last.get('i_x_seed',0):.4f} | fitness={last.get('fitness',0):.4f}
        </div>""", unsafe_allow_html=True)


# ── Experiment results ─────────────────────────────────────────────────────────
st.markdown('<div class="section-head">// PHASE 0 · METRIC VALIDATION RESULTS</div>', unsafe_allow_html=True)

runs = load_experiments()
CANONICAL = '20260427T120238Z'

if runs:
    available = list(runs.keys())
    run_labels = [runs[k]["run_name"] if runs[k]["run_name"] else k for k in available]
    default_idx = available.index(CANONICAL) if CANONICAL in available else 0
    selected_idx = st.selectbox(
        "SELECT RUN",
        range(len(available)),
        format_func=lambda i: run_labels[i],
        index=default_idx
    )
    run_id = available[selected_idx]
    data = runs[run_id]["data"]
    mean_metrics = get_mean_metrics(data)
    kw           = get_kw(data)
    effects      = get_effects(data)
    doc_count    = data.get('run', {}).get('doc_count', {})
else:
    st.caption("No experiment data found — showing canonical Phase 0 results (20260427T120238Z).")
    mean_metrics = DEMO_METRICS
    kw           = DEMO_KW
    effects      = DEMO_EFFECTS
    doc_count    = {'food': 73, 'predator': 116, 'noise': 35}


# Metric tiles
if mean_metrics:
    mt_cols = st.columns(len(mean_metrics))
    for col, (dtype, m) in zip(mt_cols, mean_metrics.items()):
        color = TYPE_COLORS.get(dtype, 'var(--green)')
        fit_color = '#00ff88' if m['fitness'] >= 0 else '#ff3c3c'
        col.markdown(f"""<div class="metric-tile" style="border-top-color:{color};">
            <div class="metric-label">{dtype.upper()} · n={m['count']}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:12px;line-height:1.9;margin-top:4px;">
                <span style="color:{color};">H</span> = {m['h_x']:.3f}<br>
                <span style="color:{color};">C</span> = {m['c_x']:.3f}<br>
                <span style="color:{color};">I</span> = {m['i_x_seed']:.4f}<br>
                <span style="color:{color};">J</span> = {m['jaccard']:.4f}<br>
                <span style="color:{fit_color};">fitness = {m['fitness']:+.4f}</span>
            </div></div>""", unsafe_allow_html=True)


# Bar chart
if mean_metrics:
    st.markdown('<div class="section-head">// METRIC COMPARISON</div>', unsafe_allow_html=True)
    types = list(mean_metrics.keys())
    metrics_to_plot = ['h_x', 'c_x', 'i_x_seed', 'jaccard']
    metric_labels   = {'h_x': 'H(X)', 'c_x': 'C(X)', 'i_x_seed': 'I(X;seed)', 'jaccard': 'Jaccard'}
    fig = go.Figure()
    for dtype in types:
        fig.add_trace(go.Bar(
            name=dtype.upper(),
            x=[metric_labels[m] for m in metrics_to_plot],
            y=[mean_metrics[dtype][m] for m in metrics_to_plot],
            marker_color=TYPE_COLORS.get(dtype, '#888'),
            marker_line_color='rgba(0,0,0,0.3)', marker_line_width=1,
        ))
    fig.update_layout(
        barmode='group', height=280,
        legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center', font=dict(color='#b0c8b8')),
        xaxis=dict(tickfont=dict(color='#b0c8b8')),
        yaxis=dict(tickfont=dict(color='#b0c8b8'), gridcolor='#0d2a1a'),
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)


# KW gauges
if kw:
    st.markdown('<div class="section-head">// KRUSKAL-WALLIS · STATISTICAL SIGNIFICANCE</div>', unsafe_allow_html=True)
    kw_labels = {'h_x': 'H(X)', 'c_x': 'C(X)', 'i_x_seed': 'I(X;seed)', 'jaccard': 'Jaccard'}
    gauge_cols = st.columns(len(kw))
    for col, (metric, res) in zip(gauge_cols, kw.items()):
        pval = res['p']
        neg_log_p = min(-np.log10(max(pval, 1e-16)), 16)
        is_sig = pval < 0.05
        bar_color = '#00ff88' if is_sig else '#ff3c3c'
        p_str = f"{pval:.2e}"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=neg_log_p,
            number={'font': {'color': bar_color, 'size': 18}},
            gauge={
                'axis': {'range': [0, 16], 'tickvals': [0, 1.3, 4, 8, 12, 16],
                         'ticktext': ['1', '0.05', '1e-4', '1e-8', '1e-12', '1e-16'],
                         'tickfont': {'color': '#3a5a46', 'size': 9}},
                'bar': {'color': bar_color}, 'bgcolor': '#060d12', 'bordercolor': '#0d2a1a',
                'steps': [{'range': [0, 1.3], 'color': 'rgba(255,60,60,0.08)'},
                          {'range': [1.3, 16], 'color': 'rgba(0,255,136,0.08)'}],
                'threshold': {'line': {'color': '#ffb300', 'width': 2}, 'value': 1.3},
            },
            title={'text': f"{kw_labels.get(metric, metric)}<br><span style='font-size:10px;color:#3a5a46'>p = {p_str}</span>",
                   'font': {'color': '#b0c8b8', 'size': 12}},
        ))
        fig.update_layout(height=200, **PLOTLY_LAYOUT)
        col.plotly_chart(fig, use_container_width=True)
        sig_text = "SIGNIFICANT" if is_sig else "NOT SIG"
        sig_kind = "green" if is_sig else "red"
        col.markdown(f'<div style="text-align:center;"><span class="badge badge-{sig_kind}">{sig_text} p={p_str}</span></div>', unsafe_allow_html=True)


# Effect sizes
if effects:
    st.markdown('<div class="section-head">// EFFECT SIZES · RANK-BISERIAL r</div>', unsafe_allow_html=True)
    eff_labels  = {'h_x': 'H(X)', 'c_x': 'C(X)', 'i_x_seed': 'I(X;seed)', 'jaccard': 'Jaccard'}
    comp_labels = {'food_vs_predator': 'food/predator', 'food_vs_noise': 'food/noise', 'predator_vs_noise': 'predator/noise'}
    comparisons = ['food_vs_predator', 'food_vs_noise', 'predator_vs_noise']
    rows = []
    for comp in comparisons:
        row = {'COMPARISON': comp_labels[comp]}
        for m, mdata in effects.items():
            val = mdata.get(comp, 0)
            row[eff_labels.get(m, m)] = f"{val:+.3f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).reset_index(drop=True), use_container_width=True, hide_index=True)
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);margin-top:6px;">
    │ |r| &gt; 0.4 = large effect &nbsp;│&nbsp; |r| &gt; 0.2 = medium &nbsp;│&nbsp; |r| &lt; 0.1 = small │
    </div>""", unsafe_allow_html=True)


# Fitness function
st.markdown('<div class="section-head">// FITNESS FUNCTION · CALIBRATED WEIGHTS</div>', unsafe_allow_html=True)
fw1, fw2, fw3, fw4 = st.columns(4)
fw1.markdown(tile_html("w1 · COMPLEXITY",      "0.3", "C(X) weight"), unsafe_allow_html=True)
fw2.markdown(tile_html("w2 · MUTUAL INFO",     "0.5", "I(X;seed) weight"), unsafe_allow_html=True)
fw3.markdown(tile_html("w3 · DISORGANIZATION", "0.2", "H_dezorg weight"), unsafe_allow_html=True)
fw4.markdown(tile_html("VALIDATION", "LOCKED", "grid search · sum=1", 'var(--blue)'), unsafe_allow_html=True)

if mean_metrics:
    fit_types  = list(mean_metrics.keys())
    fit_vals   = [mean_metrics[t]['fitness'] for t in fit_types]
    fit_colors = ['#00ff88' if v >= 0 else '#ff3c3c' for v in fit_vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[t.upper() for t in fit_types], y=fit_vals,
        marker_color=fit_colors, marker_line_color='rgba(0,0,0,0.3)', marker_line_width=1,
        text=[f"{v:+.4f}" for v in fit_vals], textposition='outside',
        textfont=dict(color='#b0c8b8', size=11),
    ))
    fig.add_hline(y=0, line_color='#3a5a46', line_dash='dash')
    fig.update_layout(
        height=220, showlegend=False,
        title=dict(text='FITNESS BY TYPE', font=dict(color='#00994d', size=11)),
        yaxis=dict(tickfont=dict(color='#b0c8b8'), gridcolor='#0d2a1a'),
        xaxis=dict(tickfont=dict(color='#b0c8b8')),
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)


# Corpus
st.markdown('<div class="section-head">// CORPUS · PHASE 0 DATA</div>', unsafe_allow_html=True)
corpus_df = pd.DataFrame([
    {'FILE': f, 'N': n, 'TYPE': dtype, 'DOMAIN': domain, 'SOURCE': src}
    for f, n, src, dtype, domain in DEMO_CORPUS
])
st.dataframe(corpus_df.reset_index(drop=True), use_container_width=True, hide_index=True)


# Config
st.markdown('<div class="section-head">// ACTIVE CONFIGURATION</div>', unsafe_allow_html=True)
cfg1, cfg2 = st.columns(2)
cfg1.markdown("""<div class="metric-tile"><div class="metric-label">MODEL</div>
<div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--text);line-height:1.8;">
base: unsloth/qwen3-8b-base-unsloth-bnb-4bit<br>backend: unsloth · 4-bit quantized<br>
temperature: 0.0 (greedy · deterministic)<br>max_new_tokens: 200<br>seed: 42</div></div>""", unsafe_allow_html=True)
cfg2.markdown("""<div class="metric-tile"><div class="metric-label">PIPELINE</div>
<div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--text);line-height:1.8;">
fitness = 0.3·C(X) + 0.5·I(X;seed) − 0.2·H_dezorg<br>jaccard: enabled (diagnostic)<br>
reproduction: disabled<br>HGT: disabled<br>cannibalism: disabled</div></div>""", unsafe_allow_html=True)

st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--dim);
letter-spacing:2px;border-top:1px solid var(--border);padding-top:12px;margin-top:24px;text-align:center;">
NEUROMANCER · EVOLUTIONARY LLM RESEARCH · UNIVERSITY OF SILESIA · KATOWICE ·
qwen3:8b-base + unsloth · RTX 4090 24GB · AMD THREADRIPPER 7960X · 256GB RAM</div>""", unsafe_allow_html=True)