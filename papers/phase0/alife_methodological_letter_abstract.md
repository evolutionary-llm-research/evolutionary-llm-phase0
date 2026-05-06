# ALife Methodological Letter: Metric Validation for Information Quality Discrimination

## Target: *Artificial Life* (Methodological Letter format, ~180–220 words)

---

## ABSTRACT

Large language models (LLMs) lack validated metrics for quantifying information quality when exposed to adversarial content (misinformation, pseudoscience). Such metrics are essential for evolutionary experiments studying population dynamics under information pressure. We validate a framework of five information-theoretic metrics—Shannon entropy H(X), effective complexity C(X), mutual information I(X;seed), Jaccard overlap, and disorganization entropy H_dezorg—against a curated corpus of 880 samples (400 peer-reviewed articles, 400 misinformation/toxin samples, 80 noise controls) stratified across five domains (climate, vaccines, alternative medicine, cancer, GMO) using qwen3:8b-base. 

**Key findings:** (1) Within-domain comparisons reveal domain-heterogeneity in information signatures—vaccines toxin exhibits high complexity but low disorganization (pseudoscientific structure), while other domains show chaotic expression. (2) Global discrimination achieves robust effect sizes for H_dezorg (r=+0.461, p<1e-29 †) and C(X) (r=−0.352, p<1e-17 †), validating a sequential diagnostic profile: complexity first, then coherence. (3) Composite fitness function (0.3·C(X) + 0.5·I(X;seed) − 0.2·H_dezorg) achieves three-class discrimination (food/toxin/noise) with very large effect size (r=−0.629 for food vs noise, p<1e-18 †), confirming robust information quality hierarchy.

**Limitations:** Single model, single inference configuration; no multi-agent population validation yet. **Significance:** Framework is reproducible and ready for Phase 1 population mechanics experiments. All code and data are open-source.

---

## NOTES FOR REVISION

- **Bonferroni correction:** α_corrected = 0.002 (per-domain), maintaining multiple comparison rigor
- **Effect size interpretation:** r ∈ [−1,1], with r≈0.3–0.5 medium, r>0.5 large
- **Vaccines anomaly:** Explicitly flag as domain heterogeneity, not a failure—this is the first publication hint
- **Reproducibility:** Mention config-driven experiment design, all RNG seeded, LoRA metadata logged
