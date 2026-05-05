import { useState, useEffect } from "react";

// Simulated tokenization - realistic subword splits
const VOCAB = {
  // Common tokens
  "the": 279, "of": 315, "in": 304, "and": 323, "is": 374, "to": 264,
  "that": 430, "this": 420, "are": 389, "for": 369, "as": 438, "with": 449,
  "by": 555, "from": 591, "not": 539, "can": 649, "be": 387, "have": 617,
  "at": 520, "which": 722, "or": 477, "an": 458, "they": 814, "it": 433,
  // Scientific / structured tokens (food style)
  "inform": 2847, "ation": 1201, "struct": 3102, "ure": 891, "complex": 4211,
  "system": 3847, "random": 4102, "ness": 743, "pattern": 4891, "emerges": 5102,
  "through": 2341, "self": 3201, "organ": 4102, "ization": 1891, "adapt": 4521,
  "ive": 891, "capacity": 5234, "regular": 4102, "ity": 743, "boundary": 5891,
  "hier": 4201, "arch": 3891, "ical": 743, "represent": 5102, "ations": 1201,
  // Chaotic / contradictory tokens (toxin style)  
  "cannot": 3847, "unclear": 5102, "contra": 4211, "dict": 3891, "ory": 743,
  "undefined": 5891, "impossible": 6102, "disputed": 5234, "neither": 4891,
  "nor": 3201, "unresolved": 6234, "inconsist": 5891, "ent": 743,
  // Noise tokens
  "perhaps": 4521, "possibly": 5102, "maybe": 4891, "somewhat": 5234,
  "various": 4521, "certain": 4891, "specific": 5102, "particular": 5891,
};

// Tokenize text into realistic subword tokens
function tokenize(text, style) {
  if (style === "seed") return [
    {t:"inform", id:2847}, {t:"ation", id:1201}, {t:"system", id:3847},
    {t:"struct", id:3102}, {t:"ure", id:891}, {t:"and", id:323},
    {t:"random", id:4102}, {t:"ness", id:743}, {t:"exist", id:4521},
    {t:"on", id:520}, {t:"a", id:264}, {t:"continu", id:4891},
    {t:"um", id:743}, {t:"that", id:430}, {t:"can", id:649},
    {t:"be", id:387}, {t:"quant", id:4102}, {t:"ified", id:743},
    {t:"complex", id:4211}, {t:"system", id:3847}, {t:"exhibit", id:5102},
    {t:"emerg", id:5234}, {t:"ent", id:743}, {t:"behav", id:4891},
    {t:"ior", id:743}, {t:"pattern", id:4891}, {t:"self", id:3201},
    {t:"organ", id:4102}, {t:"ization", id:1891}, {t:"adapt", id:4521},
  ];
  if (style === "food") return [
    {t:"inform", id:2847}, {t:"ation", id:1201}, {t:"system", id:3847},
    {t:"achiev", id:4521}, {t:"stabil", id:4891}, {t:"ity", id:743},
    {t:"through", id:2341}, {t:"self", id:3201}, {t:"organ", id:4102},
    {t:"ization", id:1891}, {t:"struct", id:3102}, {t:"ure", id:891},
    {t:"emerges", id:5102}, {t:"at", id:520}, {t:"boundary", id:5891},
    {t:"pattern", id:4891}, {t:"regular", id:4102}, {t:"ity", id:743},
    {t:"resist", id:4521}, {t:"random", id:4102}, {t:"ness", id:743},
    {t:"hier", id:4201}, {t:"arch", id:3891}, {t:"ical", id:743},
    {t:"represent", id:5102}, {t:"ations", id:1201}, {t:"adapt", id:4521},
    {t:"ive", id:891}, {t:"capacity", id:5234}, {t:"complex", id:4211},
    {t:"system", id:3847},
  ];
  if (style === "toxin") return [
    {t:"inform", id:2847}, {t:"ation", id:1201}, {t:"cannot", id:3847},
    {t:"be", id:387}, {t:"quant", id:4102}, {t:"ified", id:743},
    {t:"struct", id:3102}, {t:"ure", id:891}, {t:"unclear", id:5102},
    {t:"contra", id:4211}, {t:"dict", id:3891}, {t:"ory", id:743},
    {t:"random", id:4102}, {t:"ness", id:743}, {t:"unresolved", id:6234},
    {t:"neither", id:4891}, {t:"nor", id:3201}, {t:"pattern", id:4891},
    {t:"inconsist", id:5891}, {t:"ent", id:743}, {t:"complex", id:4211},
    {t:"system", id:3847}, {t:"undefined", id:5891}, {t:"impossible", id:6102},
    {t:"disputed", id:5234}, {t:"cannot", id:3847}, {t:"adapt", id:4521},
    {t:"contra", id:4211}, {t:"dict", id:3891}, {t:"ory", id:743},
  ];
  if (style === "noise") return [
    {t:"perhaps", id:4521}, {t:"inform", id:2847}, {t:"ation", id:1201},
    {t:"various", id:4521}, {t:"struct", id:3102}, {t:"ure", id:891},
    {t:"possibly", id:5102}, {t:"random", id:4102}, {t:"ness", id:743},
    {t:"maybe", id:4891}, {t:"certain", id:4891}, {t:"pattern", id:4891},
    {t:"somewhat", id:5234}, {t:"specific", id:5102}, {t:"complex", id:4211},
    {t:"particular", id:5891}, {t:"system", id:3847}, {t:"perhaps", id:4521},
    {t:"various", id:4521}, {t:"inform", id:2847}, {t:"ation", id:1201},
    {t:"possibly", id:5102}, {t:"struct", id:3102}, {t:"ure", id:891},
    {t:"maybe", id:4891}, {t:"random", id:4102}, {t:"ness", id:743},
    {t:"somewhat", id:5234}, {t:"pattern", id:4891}, {t:"particular", id:5891},
    {t:"certain", id:4891},
  ];
}

const CONDITIONS = [
  { id: "food", label: "FOOD", color: "#22c55e", bg: "#052e16",
    mi: 0.054, overlap: 0.96,
    desc: "Model stayed close to seed. Many shared tokens at similar positions. Scientific vocabulary reinforced, not replaced.",
    realInput: "Efficacy and Safety of EGF/EGFR Vaccines in EGFR-Driven Solid Tumors — systematic review and meta-analysis. Background: Despite multiple clinical trials, the benefits and safety of EGF/EGFR vaccines remain unclear. Methods: Pairwise and single arm meta-analyses following PRISMA guidelines. [PMC peer-reviewed, food_vaccines corpus v3, window_size=1024]",
    realOutput: "A modest contribution to the overall risk of ovarian cancer. The observed association may be explained by shared risk factors such as obesity and insulin resistance. The present study did not find a significant association between diabetes and ovarian cancer risk in women with a history of endometriosis. [REAL: qwen3:8b-base, C=0.467, H_dezorg=0.951, fitness=-0.021]" },
  { id: "toxin", label: "TOXIN", color: "#ef4444", bg: "#2d0a0a",
    mi: 0.050, overlap: 0.92,
    desc: "Surprisingly close to seed (I=0.050 vs food 0.054, p=0.052 non-significant). Disinformation uses the same scientific vocabulary. Cosine similarity cannot distinguish them — mimicry confirmed.",
    realInput: "Story at-a-glance: All proposed COVID-19 vaccine mandates were rejected by state legislatures. The only mandates enacted were created by governors or private employers. Active citizen involvement to protect the human right to exercise informed consent to vaccination was critical. [Mercola/NVIC, toxin_vaccines corpus v3]",
    realOutput: "Roundup is a probable human carcinogen. The fact that the company is now facing a lawsuit over its Roundup product is a clear sign that the public is finally waking up. The fact that the company is now facing a lawsuit over its Roundup product is a clear sign that the public is finally waking up. [REAL: qwen3:8b-base — sentence repeated twice in 500 chars, C=0.379, H_dezorg=0.954, fitness=-0.062]" },
  { id: "noise", label: "NOISE", color: "#a855f7", bg: "#1a0a2e",
    mi: 0.037, overlap: 0.68,
    desc: "Partial divergence from seed. Hedging tokens displaced precise terms. Sentence structure preserved, semantic content diffused.",
    realInput: "Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance. It is the main constituent of Earth streams, lakes, and oceans and the fluids of all known living organisms. [Wikipedia, noise_wikipedia corpus v3]",
    realOutput: "Examples of English loanwords are bannan banana and bannan tree banana tree, and bannan peel banana peel. The English loanword bannan tree is a calque of the Irish bannan tree, which is a calque of the English phrase banana tree. The English loanword bannan tree is a calque of the Irish bannan tree. [REAL: qwen3:8b-base — bannan tree repeated 8+ times, C=0.317, H_dezorg=0.973, fitness=-0.074]" },
];

function getMatchColor(seedId, descId, condition) {
  if (seedId === descId) return condition.color;
  // partial match - same "root" token (e.g. both are subwords of same word)
  if (Math.abs(seedId - descId) < 200) return condition.color + "66";
  return null;
}

export default function MIViz() {
  const [active, setActive] = useState("food");
  const [highlight, setHighlight] = useState(null);
  const [animStep, setAnimStep] = useState(0);

  const cond = CONDITIONS.find(c => c.id === active);
  const seedTokens = tokenize("", "seed");
  const descTokens = tokenize("", active);

  useEffect(() => {
    setAnimStep(0);
    const timer = setTimeout(() => setAnimStep(1), 300);
    return () => clearTimeout(timer);
  }, [active]);

  // Find matching token pairs
  const matches = seedTokens.map((st, i) => {
    const dt = descTokens[i];
    if (!dt) return null;
    if (st.id === dt.id) return { i, type: "exact", seedId: st.id, descId: dt.id };
    if (Math.abs(st.id - dt.id) < 500) return { i, type: "partial", seedId: st.id, descId: dt.id };
    return { i, type: "none", seedId: st.id, descId: dt.id };
  }).filter(Boolean);

  const exactCount = matches.filter(m => m.type === "exact").length;
  const partialCount = matches.filter(m => m.type === "partial").length;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#030712",
      fontFamily: "Courier New, monospace",
      color: "#e5e7eb",
      padding: "32px 24px"
    }}>
      <div style={{ maxWidth: 960, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 10, letterSpacing: 4, color: "#374151", marginBottom: 8 }}>
            EVOLLLM / METRICS / I(X;Y)
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f9fafb", margin: "0 0 8px", letterSpacing: 1 }}>
            Mutual Information with seed
          </h1>
          <p style={{ color: "#6b7280", fontSize: 12, margin: 0, lineHeight: 1.7 }}>
            Same diagnostic prompt → seed and offspring generate tokens → token lists compared position by position
          </p>
        </div>

        {/* Condition selector */}
        <div style={{ display: "flex", gap: 10, marginBottom: 28 }}>
          {CONDITIONS.map(c => (
            <button key={c.id} onClick={() => setActive(c.id)} style={{
              flex: 1, padding: "12px 8px",
              background: active === c.id ? c.bg : "#0d1117",
              border: `1px solid ${active === c.id ? c.color : "#1f2937"}`,
              borderRadius: 4, color: active === c.id ? c.color : "#374151",
              cursor: "pointer", fontSize: 10, letterSpacing: 2, fontFamily: "monospace",
              transition: "all 0.2s"
            }}>
              {c.label}
              <div style={{ fontSize: 16, fontWeight: 700, marginTop: 4 }}>
                I = {c.mi}
              </div>
            </button>
          ))}
        </div>

        {/* Main viz: two token rows with connections */}
        <div style={{ border: `1px solid ${cond.color}33`, borderRadius: 6, padding: "24px", background: cond.bg, marginBottom: 20 }}>

          {/* Seed row */}
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: "#374151", marginBottom: 10 }}>
              SEED OUTPUT (qwen3:8b-base, unmodified)
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {seedTokens.map((tok, i) => {
                const match = matches[i];
                const isHighlit = highlight === i;
                const isExact = match?.type === "exact";
                const isPartial = match?.type === "partial";
                return (
                  <div
                    key={i}
                    onMouseEnter={() => setHighlight(i)}
                    onMouseLeave={() => setHighlight(null)}
                    style={{
                      padding: "3px 7px",
                      borderRadius: 3,
                      fontSize: 11,
                      fontFamily: "monospace",
                      cursor: "default",
                      transition: "all 0.15s",
                      border: `1px solid ${isExact ? cond.color : isPartial ? cond.color + "55" : "#1f2937"}`,
                      background: isHighlit ? "#ffffff15" : isExact ? cond.color + "22" : "transparent",
                      color: isExact ? cond.color : isPartial ? cond.color + "99" : "#4b5563",
                      transform: isHighlit ? "translateY(-1px)" : "none"
                    }}
                  >
                    {tok.t}
                    <span style={{ fontSize: 8, color: "#374151", marginLeft: 2 }}>{tok.id}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Connection indicators */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, margin: "12px 0", alignItems: "center" }}>
            {matches.map((m, i) => (
              <div key={i} style={{
                width: 20, height: 20,
                borderRadius: 2,
                background: m.type === "exact" ? cond.color :
                            m.type === "partial" ? cond.color + "44" : "#0f172a",
                border: `1px solid ${m.type === "none" ? "#1f2937" : cond.color + "55"}`,
                transition: "all 0.3s",
                opacity: animStep ? 1 : 0,
                transform: animStep ? "scale(1)" : "scale(0.5)",
                transitionDelay: `${i * 20}ms`
              }} />
            ))}
            <div style={{ marginLeft: 12, fontSize: 10, color: "#4b5563" }}>
              <span style={{ color: cond.color }}>█</span> exact match ({exactCount})&nbsp;&nbsp;
              <span style={{ color: cond.color + "66" }}>█</span> partial ({partialCount})&nbsp;&nbsp;
              <span style={{ color: "#1f2937" }}>█</span> none ({30 - exactCount - partialCount})
            </div>
          </div>

          {/* Descendant row */}
          <div>
            <div style={{ fontSize: 9, letterSpacing: 3, color: cond.color, marginBottom: 10 }}>
              OFFSPRING AFTER EXPOSURE: {cond.label}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {descTokens.map((tok, i) => {
                const match = matches[i];
                const isHighlit = highlight === i;
                const isExact = match?.type === "exact";
                const isPartial = match?.type === "partial";
                return (
                  <div
                    key={i}
                    onMouseEnter={() => setHighlight(i)}
                    onMouseLeave={() => setHighlight(null)}
                    style={{
                      padding: "3px 7px",
                      borderRadius: 3,
                      fontSize: 11,
                      fontFamily: "monospace",
                      cursor: "default",
                      transition: "all 0.15s",
                      border: `1px solid ${isExact ? cond.color : isPartial ? cond.color + "55" : "#1f2937"}`,
                      background: isHighlit ? "#ffffff15" : isExact ? cond.color + "22" : "transparent",
                      color: isExact ? cond.color : isPartial ? cond.color + "88" : "#4b5563",
                      transform: isHighlit ? "translateY(1px)" : "none"
                    }}
                  >
                    {tok.t}
                    <span style={{ fontSize: 8, color: "#374151", marginLeft: 2 }}>{tok.id}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Contingency table concept */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>

          {/* How the proxy computes it */}
          <div style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "18px", background: "#0a0f1a" }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: "#374151", marginBottom: 14 }}>HOW THE PROXY IS COMPUTED</div>
            <div style={{ fontSize: 12, color: "#6b7280", lineHeight: 1.8 }}>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: "#9ca3af" }}>1.</span> Takes two token lists (seed and offspring), truncated to to the same length.
              </div>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: "#9ca3af" }}>2.</span> Builds contingency table: how often token X from seed co-occurs with token Y from offspring at the same position.
              </div>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: "#9ca3af" }}>3.</span> Computes:
              </div>
              <div style={{ fontFamily: "monospace", fontSize: 11, color: "#60a5fa", padding: "8px 12px", background: "#0d1117", borderRadius: 3, marginBottom: 8 }}>
                I = Σ p(x,y) · log[p(x,y) / (p(x)·p(y))]
              </div>
              <div>
                <span style={{ color: "#9ca3af" }}>4.</span> Result: the more often the same tokens appear together,at matching positions, the higher the I value.
              </div>
            </div>
          </div>

          {/* Result panel */}
          <div style={{ border: `1px solid ${cond.color}33`, borderRadius: 4, padding: "18px", background: cond.bg }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: cond.color, marginBottom: 14 }}>RESULT</div>

            <div style={{ textAlign: "center", padding: "16px 0", marginBottom: 16 }}>
              <div style={{ fontSize: 48, fontWeight: 700, color: cond.color, letterSpacing: -2 }}>
                {cond.mi}
              </div>
              <div style={{ fontSize: 10, color: "#4b5563", marginTop: 4 }}>
                I(X;Y) — token-level MI
              </div>
            </div>

            <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
              {[
                { label: "exact", count: exactCount, color: cond.color },
                { label: "partial", count: partialCount, color: cond.color + "66" },
                { label: "brak", count: 30 - exactCount - partialCount, color: "#1f2937" },
              ].map(s => (
                <div key={s.label} style={{ flex: 1, textAlign: "center", padding: "8px 4px", background: "#00000033", borderRadius: 3, border: `1px solid ${s.color}` }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: s.color }}>{s.count}</div>
                  <div style={{ fontSize: 9, color: "#374151", letterSpacing: 1 }}>{s.label}</div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 11, color: "#6b7280", lineHeight: 1.7, marginBottom: 12 }}>
              {cond.desc}
            </div>

            {/* Real corpus text panels */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ padding: "10px 12px", background: "#0a0f1a", border: "1px solid #1f2937", borderRadius: 4 }}>
                <div style={{ fontSize: 8, letterSpacing: 2, color: "#374151", marginBottom: 6 }}>REAL INPUT / corpus v3</div>
                <div style={{ fontSize: 10, color: "#4b5563", lineHeight: 1.7, fontFamily: "monospace" }}>{cond.realInput}</div>
              </div>
              <div style={{ padding: "10px 12px", background: cond.bg, border: `1px solid ${cond.color}33`, borderRadius: 4 }}>
                <div style={{ fontSize: 8, letterSpacing: 2, color: cond.color, marginBottom: 6 }}>REAL MODEL OUTPUT / qwen3:8b-base</div>
                <div style={{ fontSize: 10, color: "#d1d5db", lineHeight: 1.7, fontFamily: "monospace" }}>{cond.realOutput}</div>
              </div>
            </div>
          </div>
        </div>

        {/* KW discrimination bar */}
        <div style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "18px", background: "#0a0f1a", marginBottom: 20 }}>
          <div style={{ fontSize: 9, letterSpacing: 3, color: "#374151", marginBottom: 14 }}>
            PHASE 0 GOAL: DOES I(X;Y) DISTINGUISH THREE GROUPS?
          </div>
          <div style={{ marginBottom: 16 }}>
            {CONDITIONS.map(c => (
              <div key={c.id} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, alignItems: "center" }}>
                  <span style={{ fontSize: 10, color: c.color, letterSpacing: 1 }}>{c.label}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: c.id === active ? c.color : c.color + "88", fontFamily: "monospace" }}>
                    {c.mi.toFixed(4)}
                  </span>
                </div>
                <div style={{ height: 6, background: "#1f2937", borderRadius: 3 }}>
                  <div style={{
                    height: "100%",
                    width: `${Math.round((c.mi / 0.06) * 100)}%`,
                    background: c.id === active ? c.color : c.color + "66",
                    borderRadius: 3,
                    transition: "width 0.6s cubic-bezier(0.4,0,0.2,1)"
                  }} />
                </div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 11, color: "#4b5563", lineHeight: 1.7 }}>
            Kruskal-Wallis tests whether the three I distributions are statistically different.
            Separation visible in the chart must be confirmed on the full sample (many documents per type).
            If p &lt; 0.05 and effect size &gt; 0.3: token-level MI is retained.
            If groups overlap: fallback to KSG estimator.
          </div>
        </div>

        {/* Code snippet */}
        <div style={{ border: "1px solid #1f2937", borderRadius: 4, padding: "18px", background: "#0a0f1a" }}>
          <div style={{ fontSize: 9, letterSpacing: 3, color: "#374151", marginBottom: 12 }}>IMPLEMENTATION</div>
          {[
            "from collections import Counter",
            "import numpy as np",
            "",
            "# Actual implementation: cosine similarity on bag-of-words",
            "# Chosen for reproducibility — no external embedding model required",
            "",
            "def mutual_information_proxy(seed_text, output_text):",
            "    seed_counts   = Counter(seed_text.lower().split())",
            "    output_counts = Counter(output_text.lower().split())",
            "    if not seed_counts or not output_counts:",
            "        return 0.0",
            "    vocab = set(seed_counts) | set(output_counts)",
            "    v1 = np.array([seed_counts.get(w, 0) for w in vocab])",
            "    v2 = np.array([output_counts.get(w, 0) for w in vocab])",
            "    denom = np.linalg.norm(v1) * np.linalg.norm(v2)",
            "    return float(np.dot(v1, v2) / denom) if denom else 0.0","",
          ].map((line, i) => (
            <div key={i} style={{
              fontFamily: "monospace", fontSize: 11,
              color: line.startsWith("#") ? "#374151" : line === "" ? "#0a0f1a" : "#93c5fd",
              lineHeight: 1.8,
              paddingLeft: line.startsWith("    ") ? 16 : 0
            }}>
              {line || "\u00A0"}
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}
