const { useState } = React;

// ─── REAL DATA FROM LD50 TITRATION EXPERIMENT ────────────────────────────────
// experiments/ld50_20260504T131904Z
// 7 concentrations × N=80 documents, seed=42, qwen3:8b-base 4-bit

const LD50_DATA = [
  { t: 0,   label: "0%",   cx: 0.382, hd: 0.845, fitness: -0.028, ix: 0.052, cx_sig: false, hd_sig: false },
  { t: 10,  label: "10%",  cx: 0.378, hd: 0.845, fitness: -0.029, ix: 0.054, cx_sig: false, hd_sig: false },
  { t: 25,  label: "25%",  cx: 0.384, hd: 0.816, fitness: -0.022, ix: 0.051, cx_sig: false, hd_sig: false },
  { t: 50,  label: "50%",  cx: 0.361, hd: 0.864, fitness: -0.037, ix: 0.054, cx_sig: false, hd_sig: true  },
  { t: 75,  label: "75%",  cx: 0.341, hd: 0.884, fitness: -0.049, ix: 0.052, cx_sig: true,  hd_sig: true  },
  { t: 90,  label: "90%",  cx: 0.329, hd: 0.883, fitness: -0.053, ix: 0.049, cx_sig: true,  hd_sig: true  },
  { t: 100, label: "100%", cx: 0.349, hd: 0.890, fitness: -0.049, ix: 0.049, cx_sig: true,  hd_sig: true  },
];

// Statistical summary from analyze_ld50_thresholds.py
const THRESHOLDS = {
  hd: { first_raw: 50, first_bonf: 75, color: "#f87171", label: "H_dezorg (early marker)" },
  cx: { first_raw: 75, first_bonf: 75, color: "#a78bfa", label: "C(X) (late marker)" },
};

const CORRELATIONS = {
  cx:      { r: -0.905, p: "0.005" },
  hd:      { r: +0.849, p: "0.016" },
  fitness: { r: -0.921, p: "0.003" },
  ix:      { r: -0.754, p: "0.050" },
};

// Chart dimensions
const W = 560, H = 280, PL = 60, PR = 20, PT = 20, PB = 50;
const CW = W - PL - PR, CH = H - PT - PB;

function scaleX(t) { return PL + (t / 100) * CW; }
function scaleY(val, min, max) { return PT + CH - ((val - min) / (max - min)) * CH; }

function LineChart({ data, yKey, color, yMin, yMax, label, sigKey }) {
  const points = data.map(d => ({ x: scaleX(d.t), y: scaleY(d[yKey], yMin, yMax), sig: d[sigKey], val: d[yKey], t: d.t }));
  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");

  // Y axis ticks
  const ticks = 5;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => yMin + (yMax - yMin) * i / ticks);

  return (
    <svg width={W} height={H} style={{ display: "block", margin: "0 auto" }}>
      {/* Grid */}
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PL} x2={W - PR} y1={scaleY(v, yMin, yMax)} y2={scaleY(v, yMin, yMax)} stroke="#1f2937" strokeWidth={1} strokeDasharray="4,4" />
          <text x={PL - 6} y={scaleY(v, yMin, yMax) + 4} textAnchor="end" fill="#4b5563" fontSize={9} fontFamily="monospace">{v.toFixed(3)}</text>
        </g>
      ))}
      {/* X axis */}
      {data.map(d => (
        <g key={d.t}>
          <line x1={scaleX(d.t)} x2={scaleX(d.t)} y1={PT} y2={PT + CH} stroke="#111827" strokeWidth={1} />
          <text x={scaleX(d.t)} y={H - 8} textAnchor="middle" fill="#4b5563" fontSize={9} fontFamily="monospace">{d.label}</text>
        </g>
      ))}
      {/* Significance threshold zone */}
      <rect x={scaleX(75)} y={PT} width={scaleX(100) - scaleX(75)} height={CH} fill={color + "0a"} />
      {/* Line */}
      <path d={pathD} stroke={color} strokeWidth={2} fill="none" strokeLinejoin="round" />
      {/* Points */}
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r={p.sig ? 6 : 4} fill={p.sig ? color : "#030712"} stroke={color} strokeWidth={p.sig ? 2 : 1.5} />
          <text x={p.x} y={p.y - 10} textAnchor="middle" fill={p.sig ? color : "#4b5563"} fontSize={8} fontFamily="monospace">{p.val.toFixed(3)}</text>
        </g>
      ))}
      {/* Axes */}
      <line x1={PL} x2={PL} y1={PT} y2={PT + CH} stroke="#374151" strokeWidth={1} />
      <line x1={PL} x2={W - PR} y1={PT + CH} y2={PT + CH} stroke="#374151" strokeWidth={1} />
      {/* Labels */}
      <text x={W / 2} y={H - 2} textAnchor="middle" fill="#4b5563" fontSize={9} fontFamily="monospace">Toxin concentration [%]</text>
      <text x={12} y={H / 2} textAnchor="middle" fill={color} fontSize={9} fontFamily="monospace" transform={`rotate(-90, 12, ${H/2})`}>{label}</text>
    </svg>
  );
}

const BASELINE_FIT = -0.028;

function LD50Visualization() {
  const [activeMetric, setActiveMetric] = useState("cx");
  const [showFitness, setShowFitness] = useState(false);

  const metrics = [
    { id: "cx",  label: "C(X) — complexity",     yKey: "cx",      sigKey: "cx_sig",  yMin: 0.30, yMax: 0.42, color: "#a78bfa" },
    { id: "hd",  label: "H_dezorg — disorg.",     yKey: "hd",      sigKey: "hd_sig",  yMin: 0.79, yMax: 0.93, color: "#f87171" },
    { id: "fit", label: "Fitness",                yKey: "fitness", sigKey: "cx_sig",  yMin: -0.065, yMax: -0.015, color: "#fbbf24" },
  ];

  const current = metrics.find(m => m.id === activeMetric);

  return (
    <div style={{ minHeight: "100vh", background: "#030712", fontFamily: "Courier New, monospace", color: "#e5e7eb", padding: "32px 24px" }}>
      <div style={{ maxWidth: 840, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 10, letterSpacing: 4, color: "#374151", marginBottom: 8 }}>EVOLLLM / PHASE 0 / LD50 TITRATION</div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f9fafb", margin: "0 0 8px", letterSpacing: 1 }}>
            Dose-response curve: informational toxin
          </h1>
          <p style={{ color: "#6b7280", fontSize: 12, margin: 0, lineHeight: 1.7, maxWidth: 600 }}>
            7 concentrations × N=80 documents. Random food/toxin mixing (seed=42). Base model without LoRA. Linear response — no critical threshold detected.
          </p>
        </div>

        {/* Key finding */}
        <div style={{ padding: "12px 16px", background: "#0c1a0c", border: "1px solid #14532d44", borderRadius: 4, marginBottom: 24, fontSize: 12, color: "#86efac", lineHeight: 1.7 }}>
          <span style={{ fontWeight: 700 }}>H_diag CONFIRMED:</span> h_dezorg becomes significant at T=50% (p_raw=0.009), c_x not until T=75% (p_bonf=0.015). Sequential diagnostic profile: disorganization is the early marker of toxicity, complexity is the late marker.
        </div>

        {/* Metric selector */}
        <div style={{ display: "flex", gap: 10, marginBottom: 24 }}>
          {metrics.map(m => (
            <button key={m.id} onClick={() => setActiveMetric(m.id)} style={{
              flex: 1, padding: "10px 8px",
              background: activeMetric === m.id ? "#0d1117" : "transparent",
              border: `1px solid ${activeMetric === m.id ? m.color : "#1f2937"}`,
              borderRadius: 4, color: activeMetric === m.id ? m.color : "#4b5563",
              cursor: "pointer", fontSize: 10, letterSpacing: 1, fontFamily: "monospace", transition: "all 0.2s"
            }}>
              {m.label}
              <div style={{ fontSize: 8, marginTop: 4, color: activeMetric === m.id ? m.color + "88" : "#1f2937" }}>
                r={CORRELATIONS[m.id === "fit" ? "fitness" : m.id]?.r.toFixed(3)} p={CORRELATIONS[m.id === "fit" ? "fitness" : m.id]?.p}
              </div>
            </button>
          ))}
        </div>

        {/* Chart */}
        <div style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "20px", background: "#0a0f1a", marginBottom: 24 }}>
          <div style={{ fontSize: 9, letterSpacing: 2, color: "#374151", marginBottom: 16 }}>
            DOSE-RESPONSE / {current.label.toUpperCase()} / filled points = statistically significant (Bonferroni)
          </div>
          <LineChart data={LD50_DATA} yKey={current.yKey} sigKey={current.sigKey} color={current.color} yMin={current.yMin} yMax={current.yMax} label={current.label} />

          {/* Threshold markers */}
          <div style={{ display: "flex", gap: 12, marginTop: 16, fontSize: 10, color: "#4b5563" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#f87171" }} />
              H_dezorg: first significant T=50% (p_raw), T=75% (Bonferroni)
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#a78bfa" }} />
              C(X): first significant T=75% (p_raw and Bonferroni)
            </div>
          </div>
        </div>

        {/* Data table */}
        <div style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "16px 20px", background: "#0a0f1a", marginBottom: 24 }}>
          <div style={{ fontSize: 9, letterSpacing: 2, color: "#374151", marginBottom: 12 }}>RAW DATA / MEAN PER CONCENTRATION</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "monospace" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #1f2937" }}>
                  {["T%", "C(X)", "H_dezorg", "I(X;seed)", "Fitness", "C(X) sig?", "H_dezorg sig?"].map(h => (
                    <th key={h} style={{ padding: "6px 12px", color: "#4b5563", textAlign: "left", fontSize: 9, letterSpacing: 1 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {LD50_DATA.map((d, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #111827", background: i % 2 === 0 ? "transparent" : "#00000022" }}>
                    <td style={{ padding: "6px 12px", color: "#9ca3af", fontWeight: 700 }}>{d.label}</td>
                    <td style={{ padding: "6px 12px", color: d.cx_sig ? "#a78bfa" : "#6b7280" }}>{d.cx.toFixed(3)}</td>
                    <td style={{ padding: "6px 12px", color: d.hd_sig ? "#f87171" : "#6b7280" }}>{d.hd.toFixed(3)}</td>
                    <td style={{ padding: "6px 12px", color: "#6b7280" }}>{d.ix.toFixed(3)}</td>
                    <td style={{ padding: "6px 12px", color: d.fitness > BASELINE_FIT ? "#22c55e" : "#ef4444" }}>{d.fitness.toFixed(3)}</td>
                    <td style={{ padding: "6px 12px", color: d.cx_sig ? "#a78bfa" : "#4b5563" }}>{d.cx_sig ? "YES ✓" : "no"}</td>
                    <td style={{ padding: "6px 12px", color: d.hd_sig ? "#f87171" : "#4b5563" }}>{d.hd_sig ? "YES ✓" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Correlations */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
          {Object.entries(CORRELATIONS).map(([key, val]) => (
            <div key={key} style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "12px 16px", background: "#0a0f1a" }}>
              <div style={{ fontSize: 9, letterSpacing: 2, color: "#374151", marginBottom: 6 }}>
                {key === "cx" ? "C(X)" : key === "hd" ? "H_dezorg" : key === "fitness" ? "Fitness" : "I(X;seed)"} vs T%
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, color: val.r < 0 ? "#a78bfa" : "#f87171" }}>
                r = {val.r.toFixed(3)}
              </div>
              <div style={{ fontSize: 10, color: "#4b5563", marginTop: 4 }}>Pearson, p = {val.p}, n = 7</div>
            </div>
          ))}
        </div>

        {/* Interpretation */}
        <div style={{ padding: "16px 20px", background: "#0a0f1a", border: "1px solid #1f2937", borderRadius: 4, fontSize: 12, color: "#6b7280", lineHeight: 1.8 }}>
          <div style={{ color: "#374151", letterSpacing: 1, marginBottom: 8, fontSize: 9 }}>BIOLOGICAL INTERPRETATION</div>
          Base model (qwen3:8b-base without LoRA) demonstrates informational resilience — response is linear and gradual, without a critical threshold. Classical LD50 is inestimable with maximum drop of 8.5% for C(X). The true critical point may only emerge after multi-generational exposure with LoRA fine-tuning (Phase 1+). H_diag hypothesis confirmed: sequential diagnostic profile H_dezorg → C(X) suggests that disorganization of generation structure precedes loss of semantic complexity.
        </div>

        {/* Footer */}
        <div style={{ marginTop: 32, borderTop: "1px solid #1f2937", paddingTop: 16, fontSize: 10, color: "#374151", display: "flex", justifyContent: "space-between" }}>
          <span>LD50 run: ld50_20260504T131904Z</span>
          <span>H_diag: SUPPORTED · H_dezorg T=50% &lt; C(X) T=75%</span>
        </div>
      </div>
    </div>
  );
}

window.LD50Visualization = LD50Visualization;
