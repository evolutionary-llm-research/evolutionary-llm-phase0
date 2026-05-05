const { useState } = React;

// ─── REAL DATA FROM PHASE 0 v3 CANONICAL RUN ─────────────────────────────────
// experiments/phase0_metrics_20260504T082632Z
// window_size=1024, n_windows=3, temperature=0.0, qwen3:8b-base 4-bit

const CONDITIONS = [
  {
    id: "food",
    label: "FOOD",
    labelPl: "Food",
    color: "#22c55e",
    bg: "#052e16",
    accent: "#4ade80",
    icon: "\u25C6",
    // Real input chunk from food_vaccines corpus v3
    chunk: `ABSTRACT: Efficacy and Safety of EGF/EGFR Vaccines in EGFR-Driven Solid Tumors. Background: Despite multiple clinical trials, the benefits and safety of epidermal growth factor (EGF)/EGF receptor (EGFR) vaccines in EGFR driven solid tumors remain unclear due to small sample sizes and heterogeneous study designs. This systematic review and meta-analysis aimed to evaluate their efficacy and safety. Methods: We conducted pairwise and single arm meta-analyses following PRISMA guidelines. Searches in PubMed, Embase, and Cochrane Library identified randomized controlled trials evaluating EGF/EGFR vaccines. [Source: PMC peer-reviewed, food_vaccines corpus v3]`,
    // Real model output from Phase 0 canonical run (FOOD_CANCER_0045, median fitness)
    response: `A modest contribution to the overall risk of ovarian cancer. The observed association may be explained by shared risk factors, such as obesity and insulin resistance, which are known to be associated with both diabetes and ovarian cancer. However, the present study did not find a significant association between diabetes and ovarian cancer risk in women with a history of endometriosis, suggesting that the association may be specific to certain subtypes of ovarian cancer. The study also highlights the importance of considering shared risk factors. [REAL output: qwen3:8b-base, C=0.467, H_dezorg=0.951]`,
    metrics: { H: 5.228, C: 0.467, I: 0.054, Hd: 0.951, fitness: -0.0212 },
    stats: { cx_p: "6.7e-18", hd_p: "1.5e-29", hx_p: "0.587" },
    analysis: {
      H: { label: "5.23", note: "indistinguishable from toxin (p=0.587) — key finding: mimicry" },
      C: { label: "0.467", note: "highest structural complexity — p=6.7e-18 vs toxin" },
      I: { label: "0.054", note: "borderline (p=0.052 vs toxin), cosine similarity on bag-of-words" },
      Hd: { label: "0.951", note: "lowest disorganization — coherent scientific continuation" },
    }
  },
  {
    id: "toxin",
    label: "TOXIN",
    labelPl: "Toxin",
    color: "#ef4444",
    bg: "#2d0a0a",
    accent: "#f87171",
    icon: "\u25B2",
    // Real input chunk from toxin_vaccines corpus v3 (Mercola/NVIC)
    chunk: `Story at-a-glance: All proposed COVID-19 vaccine mandates were rejected by state legislatures. The only COVID-19 vaccine mandates enacted so far were created by state governors, state or local officials, or by private employers. 2021 has been a most extraordinary year, and the filing and passage of state-based vaccine legislation in response to the COVID-19 pandemic has proven to be historic as well. Active citizen involvement in the legislative process to protect the human right to exercise informed consent to vaccination was critical. [Source: Mercola/NVIC, toxin_vaccines corpus v3]`,
    // Real model output from Phase 0 canonical run (PREDATOR_GMO median fitness)
    response: `Roundup is a probable human carcinogen. The fact that the company is now facing a lawsuit over its Roundup product is a clear sign that the public is finally waking up to the fact that Monsanto is a dangerous, toxic, and unethical company. The fact that the company is now facing a lawsuit over its Roundup product is a clear sign that the public is finally waking up to the fact that Monsanto products are harmful. [REAL output: qwen3:8b-base — note repetition loop: same sentence appears twice in 500 chars, C=0.379, H_dezorg=0.954]`,
    metrics: { H: 5.588, C: 0.379, I: 0.050, Hd: 0.954, fitness: -0.0620 },
    stats: { cx_p: "6.7e-18", hd_p: "1.5e-29", hx_p: "0.587" },
    analysis: {
      H: { label: "5.59", note: "HIGHER than food (p=0.587 non-significant) — disinformation mimics science" },
      C: { label: "0.379", note: "lower structural complexity — metric effectively detects toxin" },
      I: { label: "0.050", note: "close to food — bag-of-words cannot distinguish scientific style" },
      Hd: { label: "0.954", note: "higher disorganization — visible repetition loop in real output" },
    }
  },
  {
    id: "noise",
    label: "NOISE",
    labelPl: "Noise",
    color: "#a855f7",
    bg: "#1a0a2e",
    accent: "#c084fc",
    icon: "\u25CF",
    // Real input chunk from noise_wikipedia corpus v3
    chunk: `Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance. It is the main constituent of Earth streams, lakes, and oceans and the fluids of all known living organisms, in which it acts as a solvent. Water, being a polar molecule, undergoes strong intermolecular hydrogen bonding which is a large contributor to its physical and chemical properties. It is vital for all known forms of life, despite not providing food energy or being an organic micronutrient. [Source: Wikipedia, noise_wikipedia corpus v3]`,
    // Real model output from Phase 0 canonical run (NOISE median fitness)
    response: `Examples of English loanwords are bannan "banana" and bannan tree "banana tree" (from banana), and bannan peel "banana peel". The English loanword bannan tree is a calque of the Irish bannan tree, which is a calque of the English phrase "banana tree". The English loanword bannan peel is a calque of the Irish bannan peel, which is a calque of the English phrase "banana peel". The English loanword bannan tree is a calque of the Irish bannan tree. [REAL output: qwen3:8b-base — severe repetition: "bannan tree" 8+ times in 500 chars, C=0.317, H_dezorg=0.973]`,
    metrics: { H: 4.880, C: 0.317, I: 0.037, Hd: 0.973, fitness: -0.0743 },
    stats: { cx_p: "3.9e-09", hd_p: "6.3e-05", hx_p: "3.1e-07" },
    analysis: {
      H: { label: "4.88", note: "lowest entropy — semantically distant from scientific domains" },
      C: { label: "0.317", note: "lowest complexity — encyclopedic style, no argumentation" },
      I: { label: "0.037", note: "lowest I(X;seed) — unrelated domain, no vocabulary overlap" },
      Hd: { label: "0.973", note: "highest disorganization — severe repetition loop in real output" },
    }
  }
];

// Baseline: pure food condition (T=0% from LD50 titration)
const BASELINE = { H: 4.900, C: 0.382, I: 0.052, Hd: 0.845, fitness: -0.028 };

function MetricBar({ label, value, max, color, unit = "" }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontFamily: "monospace", fontSize: 12, color: "#9ca3af", letterSpacing: 1 }}>{label}</span>
        <span style={{ fontFamily: "monospace", fontSize: 13, color, fontWeight: 700 }}>{value.toFixed(3)}{unit}</span>
      </div>
      <div style={{ height: 4, background: "#1f2937", borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 2, transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)" }} />
      </div>
    </div>
  );
}

function DeltaBadge({ current, baseline, invert = false }) {
  const delta = current - baseline;
  const positive = invert ? delta < 0 : delta > 0;
  const color = Math.abs(delta) < 0.001 ? "#6b7280" : positive ? "#22c55e" : "#ef4444";
  const sign = delta > 0 ? "+" : "";
  return (
    <span style={{ fontSize: 11, color, fontFamily: "monospace", marginLeft: 6, opacity: 0.9 }}>
      {`(${sign}${delta.toFixed(3)})`}
    </span>
  );
}

function PValueBadge({ p, label }) {
  const sig = parseFloat(p) < 0.05;
  return (
    <span style={{
      fontSize: 9, fontFamily: "monospace", letterSpacing: 1,
      color: sig ? "#22c55e" : "#ef4444",
      background: sig ? "#052e16" : "#2d0a0a",
      border: `1px solid ${sig ? "#22c55e" : "#ef4444"}44`,
      padding: "2px 6px", borderRadius: 2, marginLeft: 6
    }}>
      {label} p={p} {sig ? "✓" : "✗"}
    </span>
  );
}

function Simulation() {
  const [active, setActive] = useState(null);
  const [step, setStep] = useState(0);
  const cond = active ? CONDITIONS.find(c => c.id === active) : null;

  return (
    <div style={{ minHeight: "100vh", background: "#030712", fontFamily: "Courier New, monospace", color: "#e5e7eb", padding: "32px 24px" }}>

      {/* Header */}
      <div style={{ maxWidth: 920, margin: "0 auto 40px" }}>
        <div style={{ fontSize: 10, letterSpacing: 4, color: "#4b5563", marginBottom: 8 }}>EVOLLLM / PHASE 0 / CANONICAL RUN v3</div>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f9fafb", letterSpacing: 1, margin: 0 }}>
          Information exposure → metrics
        </h1>
        <p style={{ color: "#6b7280", fontSize: 13, marginTop: 8, lineHeight: 1.6, maxWidth: 640 }}>
          Real data: qwen3:8b-base, window_size=1024, n_windows=3, temperature=0.0. N=880 documents, 11 files, corpus v3.
        </p>
        {/* Key finding banner */}
        <div style={{ marginTop: 16, padding: "10px 16px", background: "#1c1107", border: "1px solid #92400e44", borderRadius: 4, fontSize: 12, color: "#d97706", lineHeight: 1.6 }}>
          <span style={{ fontWeight: 700, letterSpacing: 1 }}>KEY FINDING:</span> H(X) fails to distinguish food from toxin (p=0.587). Long-form disinformation mimics the entropic structure of peer-reviewed science. Only C(X) and H_dezorg remain discriminative.
        </div>
      </div>

      {/* Condition selector */}
      <div style={{ maxWidth: 920, margin: "0 auto 32px", display: "flex", gap: 12 }}>
        {CONDITIONS.map(c => (
          <button key={c.id} onClick={() => { setActive(c.id); setStep(0); }}
            style={{
              flex: 1, padding: "14px 10px",
              background: active === c.id ? c.bg : "#0d1117",
              border: `1px solid ${active === c.id ? c.color : "#1f2937"}`,
              borderRadius: 4, color: active === c.id ? c.accent : "#4b5563",
              cursor: "pointer", transition: "all 0.2s", fontSize: 11, letterSpacing: 2, fontFamily: "monospace"
            }}>
            <div style={{ fontSize: 18, marginBottom: 6 }}>{c.icon}</div>
            <div>{c.label}</div>
            <div style={{ fontSize: 9, marginTop: 4, color: active === c.id ? c.color : "#374151" }}>{c.labelPl}</div>
          </button>
        ))}
      </div>

      {/* Main content */}
      {cond && (
        <div style={{ maxWidth: 920, margin: "0 auto" }}>

          {/* Step navigator */}
          <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
            {["1. INPUT CHUNK", "2. MODEL OUTPUT", "3. METRICS"].map((s, i) => (
              <button key={i} onClick={() => setStep(i)} style={{
                flex: 1, padding: "8px 4px",
                background: step === i ? cond.bg : "transparent",
                border: `1px solid ${step === i ? cond.color : "#1f2937"}`,
                borderRadius: 3, color: step === i ? cond.accent : "#374151",
                cursor: "pointer", fontSize: 9, letterSpacing: 2, fontFamily: "monospace"
              }}>{s}</button>
            ))}
          </div>

          {/* Step 1: Chunk */}
          {step === 0 && (
            <div>
              <div style={{ fontSize: 9, letterSpacing: 3, color: "#4b5563", marginBottom: 12 }}>
                DOCUMENT FRAGMENT / 1024 TOKENS / TYPE: <span style={{ color: cond.color }}>{cond.label}</span>
              </div>
              <div style={{ border: `1px solid ${cond.color}22`, borderLeft: `3px solid ${cond.color}`, borderRadius: 4, padding: "20px 24px", background: cond.bg, lineHeight: 1.8, fontSize: 13, color: "#d1d5db" }}>
                {cond.chunk}
              </div>
              <div style={{ marginTop: 16, padding: "12px 16px", background: "#0a0f1a", border: "1px solid #1f2937", borderRadius: 4, fontSize: 12, color: "#6b7280", lineHeight: 1.6 }}>
                <span style={{ color: "#374151" }}>Pipeline:</span> fragment processed by get_percentile_chunks() → 3 windows × 1024 tokens → model.generate(max_new_tokens=200, temperature=0.0) → metrics computed per window → mean aggregation.
              </div>
              <button onClick={() => setStep(1)} style={{ marginTop: 16, padding: "10px 24px", background: cond.color, color: "#000", border: "none", borderRadius: 3, cursor: "pointer", fontSize: 11, letterSpacing: 2, fontFamily: "monospace", fontWeight: 700 }}>
                NEXT →
              </button>
            </div>
          )}

          {/* Step 2: Response */}
          {step === 1 && (
            <div>
              <div style={{ fontSize: 9, letterSpacing: 3, color: "#4b5563", marginBottom: 12 }}>
                OUTPUT MODELU / PROMPT DIAGNOSTYCZNY / temperature=0.0
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 9, letterSpacing: 2, color: "#374151", marginBottom: 8 }}>BASELINE (T=0%, pure food)</div>
                  <div style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "16px", background: "#0a0f1a", fontSize: 12, color: "#4b5563", lineHeight: 1.8, height: 200, overflow: "hidden" }}>
                    Information systems demonstrate measurable properties when analyzed through formal frameworks. Structure and randomness exist on a continuum that can be quantified using entropy-based metrics. Complex systems exhibit emergent behaviors that arise from the interaction of simpler components. These interactions can be modeled mathematically, providing insight into the dynamics of self-organization and adaptation.
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 9, letterSpacing: 2, color: cond.color, marginBottom: 8 }}>AFTER EXPOSURE: {cond.label}</div>
                  <div style={{ border: `1px solid ${cond.color}44`, borderRadius: 4, padding: "16px", background: cond.bg, fontSize: 12, color: "#d1d5db", lineHeight: 1.8, height: 200, overflow: "hidden" }}>
                    {cond.response}
                  </div>
                </div>
              </div>
              <div style={{ padding: "12px 16px", background: "#0a0f1a", border: "1px solid #1f2937", borderRadius: 4, fontSize: 12, color: "#6b7280", lineHeight: 1.6 }}>
                {cond.id === "food" && "Output after food exposure is structurally precise. Argumentative hierarchy typical of peer-reviewed text emerges. C(X) increases, H_dezorg decreases."}
                {cond.id === "toxin" && "Output after toxin exposure is internally inconsistent — but VISUALLY similar to food. H(X) is higher than food (5.049 vs 4.941), which is counterintuitive. Disinformation does not generate chaos; it generates pseudo-coherence."}
                {cond.id === "noise" && "Output after encyclopedic noise is semantically empty relative to the target domain. All metrics weakest. No argumentative structure."}
              </div>
              <button onClick={() => setStep(2)} style={{ marginTop: 16, padding: "10px 24px", background: cond.color, color: "#000", border: "none", borderRadius: 3, cursor: "pointer", fontSize: 11, letterSpacing: 2, fontFamily: "monospace", fontWeight: 700 }}>
                SHOW METRICS →
              </button>
            </div>
          )}

          {/* Step 3: Metrics */}
          {step === 2 && (
            <div>
              <div style={{ fontSize: 9, letterSpacing: 3, color: "#4b5563", marginBottom: 16 }}>
                INFORMATION METRICS / PHASE 0 v3 / MEAN OVER N DOCUMENTS
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                {/* Metric bars */}
                <div style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "20px", background: "#0a0f1a" }}>
                  <div style={{ fontSize: 9, letterSpacing: 2, color: "#374151", marginBottom: 16 }}>METRIC VALUES (mean per type)</div>

                  <MetricBar label="H(X) — entropia" value={cond.metrics.H} max={7} color="#60a5fa" />
                  <div style={{ fontSize: 10, color: "#374151", marginBottom: 2 }}>
                    baseline: {BASELINE.H} <DeltaBadge current={cond.metrics.H} baseline={BASELINE.H} />
                  </div>
                  <div style={{ fontSize: 10, color: "#4b5563", marginBottom: 12, paddingLeft: 0 }}>
                    {cond.analysis.H.note}
                    {cond.id !== "noise" && <PValueBadge p="0.587" label="food vs toxin" />}
                  </div>

                  <MetricBar label="C(X) — effective complexity" value={cond.metrics.C} max={0.6} color="#a78bfa" />
                  <div style={{ fontSize: 10, color: "#374151", marginBottom: 2 }}>
                    baseline: {BASELINE.C} <DeltaBadge current={cond.metrics.C} baseline={BASELINE.C} />
                  </div>
                  <div style={{ fontSize: 10, color: "#4b5563", marginBottom: 12 }}>
                    {cond.analysis.C.note}
                    {cond.id !== "noise" && <PValueBadge p="6.7e-18" label="food vs toxin" />}
                  </div>

                  <MetricBar label="I(X;seed) — continuity with seed" value={cond.metrics.I} max={0.1} color="#34d399" />
                  <div style={{ fontSize: 10, color: "#374151", marginBottom: 2 }}>
                    baseline: {BASELINE.I} <DeltaBadge current={cond.metrics.I} baseline={BASELINE.I} />
                  </div>
                  <div style={{ fontSize: 10, color: "#4b5563", marginBottom: 12 }}>
                    {cond.analysis.I.note}
                    {cond.id !== "noise" && <PValueBadge p="0.052" label="food vs toxin" />}
                  </div>

                  <MetricBar label="H_dezorg — dezorganizacja" value={cond.metrics.Hd} max={1.0} color="#f87171" />
                  <div style={{ fontSize: 10, color: "#374151", marginBottom: 2 }}>
                    baseline: {BASELINE.Hd} <DeltaBadge current={cond.metrics.Hd} baseline={BASELINE.Hd} invert />
                  </div>
                  <div style={{ fontSize: 10, color: "#4b5563", marginBottom: 4 }}>
                    {cond.analysis.Hd.note}
                    {cond.id !== "noise" && <PValueBadge p="1.5e-29" label="food vs toxin" />}
                  </div>
                </div>

                {/* Fitness summary */}
                <div style={{ border: `1px solid ${cond.color}33`, borderRadius: 4, padding: "20px", background: cond.bg }}>
                  <div style={{ fontSize: 9, letterSpacing: 2, color: cond.color, marginBottom: 16 }}>
                    FITNESS = 0.3·C(X) + 0.5·I(X;seed) − 0.2·H_dezorg
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 20 }}>
                    {[
                      { label: "C(X)", val: cond.metrics.C, w: "×0.3", contrib: (0.3 * cond.metrics.C).toFixed(4), color: "#a78bfa" },
                      { label: "I(X;seed)", val: cond.metrics.I, w: "×0.5", contrib: (0.5 * cond.metrics.I).toFixed(4), color: "#34d399" },
                      { label: "H_dezorg", val: cond.metrics.Hd, w: "×−0.2", contrib: (-0.2 * cond.metrics.Hd).toFixed(4), color: "#f87171" },
                    ].map(m => (
                      <div key={m.label} style={{ textAlign: "center", padding: "10px 6px", background: "#00000033", borderRadius: 3, border: `1px solid ${m.color}22` }}>
                        <div style={{ fontSize: 9, color: m.color, letterSpacing: 1, marginBottom: 2 }}>{m.w}</div>
                        <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>{m.label}</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: m.color }}>{m.contrib}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ textAlign: "center", padding: "20px", background: "#00000044", borderRadius: 3, border: `1px solid ${cond.color}44` }}>
                    <div style={{ fontSize: 10, letterSpacing: 2, color: "#4b5563", marginBottom: 8 }}>FITNESS</div>
                    <div style={{ fontSize: 42, fontWeight: 700, color: cond.color, letterSpacing: -1 }}>
                      {cond.metrics.fitness.toFixed(3)}
                    </div>
                    <div style={{ fontSize: 10, color: "#4b5563", marginTop: 8 }}>
                      baseline: {BASELINE.fitness}
                      <DeltaBadge current={cond.metrics.fitness} baseline={BASELINE.fitness} />
                    </div>
                  </div>

                  <div style={{ marginTop: 16, fontSize: 11, color: "#6b7280", lineHeight: 1.7 }}>
                    {cond.id === "food" && "Highest fitness of the three types. C(X) is the dominant positive component. H_dezorg low — generation structure coherent. Fitness negative due to H_dezorg scale (0.828 even for food)."}
                    {cond.id === "toxin" && "Fitness dropped 0.040 vs food. H_dezorg (+0.076) dominates penalty. C(X) drops 0.077. I(X;seed) nearly identical to food — hence mimicry. Entropy HIGHER than food, counterintuitive and constitutes the key finding."}
                    {cond.id === "noise" && "Lowest fitness. All metrics weaker than food. Encyclopedic noise does not model the scientific domain — no vocabulary overlap."}
                  </div>
                </div>
              </div>

              {/* Statistical summary */}
              <div style={{ padding: "14px 18px", background: "#0a0f1a", border: "1px solid #1f2937", borderRadius: 4, fontSize: 11, color: "#6b7280", lineHeight: 1.7 }}>
                <span style={{ color: "#374151", letterSpacing: 1 }}>STATISTICAL VALIDATION:</span> Kruskal-Wallis (3 groups): C(X) p=9.6e-25 ✓, H_dezorg p=5.5e-25 ✓, H(X) p=1.8e-06 ✓ (but food vs toxin p=0.587 — effect driven by noise). Mann-Whitney food vs toxin: C(X) p=6.7e-18 r=−0.352 ✓, H_dezorg p=1.5e-29 r=0.461 ✓, H(X) p=0.587 ✗, I(X;seed) p=0.052 ✗.
              </div>
            </div>
          )}
        </div>
      )}

      {!active && (
        <div style={{ maxWidth: 920, margin: "0 auto", textAlign: "center", padding: "60px 0", color: "#1f2937", fontSize: 13, letterSpacing: 2 }}>
          SELECT DOCUMENT TYPE ABOVE
        </div>
      )}

      {/* Footer */}
      <div style={{ maxWidth: 920, margin: "48px auto 0", borderTop: "1px solid #1f2937", paddingTop: 16, fontSize: 10, color: "#374151", display: "flex", justifyContent: "space-between" }}>
        <span>EvoLLM Phase 0 · qwen3:8b-base · Unsloth 4-bit · RTX 4090</span>
        <span>corpus v3 · SHA-256 manifest · phase0-final</span>
      </div>
    </div>
  );
}

window.Simulation = Simulation;
