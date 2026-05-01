# Run Naming Convention — Neuromancer Pipeline

Format: {phase}_{type}_{number}

## Phase prefixes
- p0: Phase 0 (metric validation)
- p1: Phase 1 (information ecology, single model)
- p2: Phase 2 (evolutionary dynamics, population)
- p3: Phase 3 (HGT + cannibalism)
- p4: Phase 4 (emergent archetypes)

## Type tags
### Phase 0
- validation: canonical metric validation run
- sensitivity: token length sensitivity analysis
- corpus: per-dataset corpus quality analysis
- grid: fitness weight grid search

### Phase 1
- savanna: savanna biome (60% food / 20% predator / 20% noise, K=30)
- desert: desert biome (10% food / 70% predator / 20% noise, K=10)
- plain: plain biome (80% food / 5% predator / 15% noise, K=25)

### Phase 2
- evo_id: Id archetype evolution
- evo_ego: Ego archetype evolution
- evo_super: Superego archetype evolution
- evo_pop: mixed population

### Phase 3
- hgt: horizontal gene transfer experiment
- cannibal: cannibalism experiment

## Number
- Sequential integer starting from 01
- Increment when rerunning same experiment type

## Examples
p0_validation_01    ← canonical Phase 0 run
p0_sensitivity_01   ← sensitivity analysis
p0_corpus_01        ← corpus quality analysis
p1_savanna_01       ← first savanna biome run
p1_savanna_02       ← second savanna biome run (different params)
p2_evo_id_01        ← first Id archetype evolution
p3_hgt_01           ← first HGT experiment

## Usage
python -m src.analysis.phase0_metric_validation \
  --config config/phase0_metrics_validation.yaml \
  --run-name p0_validation_02
