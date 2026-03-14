# Power-118 Constraint-Aware Hybrid

This document summarizes the current generations of `power-118` hybrid solving and how they relate to the "Constraint Matters" direction.

## Generations

### v1
- warm-start only
- ML predicts commitment and dispatch
- exact solver receives a starting point
- safe but usually does not reduce the search space enough

### v2
- fixing-based constraint-aware hybrid
- exact samples provide active or tight constraint labels
- a lightweight fixing head predicts which commitment binaries can be fixed safely
- reduced solve is still conservative and falls back to warm-start or exact when needed

### v3
- structured constraint representation plus constraint scoring
- each candidate constraint gets:
  - instance-level features
  - abstract-level features
  - active or tight or rank labels
- the hybrid solver can use scoring-driven staged reduction:
  - keep top-ranked critical constraints
  - defer lower-priority ramp or line constraints
  - reactivate violated deferred constraints
  - fall back safely when reduction is rejected

## Relation To "Constraint Matters"

What is now closer to the paper:
- explicit separation between instance-level and abstract-level features
- constraint-level candidate records rather than only case-level summaries
- constraint scoring and ranking instead of only solution prediction
- staged reduction and add-back logic rather than only warm-starts

What is still not implemented:
- graph neural networks
- cross-modal attention between instance and abstract representations
- learned abstract-level embeddings over large MILP families
- end-to-end multimodal graph reduction architecture

This means the current implementation is a structured engineering approximation inspired by the paper, not a faithful reproduction of the full method.

## Data Products

Dataset build now produces:
- `features`
- `targets`
- `constraint_labels`
- `fixing_labels`
- `constraint_candidates`
- `metadata`

Constraint candidate rows include:
- `sampleId`
- `constraintId`
- `constraintType`
- instance-level feature columns with `inst_` prefix
- abstract-level feature columns with `abs_` prefix
- `labelActive`
- `labelTight`
- `labelRankScore`
- `canBeReduced`
- `canBeDeferred`
- `canBeFixed`

## Training Products

Training now supports:
- schedule prediction heads
- fixing-mask head
- constraint scoring head

Metadata includes:
- `constraintRepresentationVersion`
- `constraintScoringModelEnabled`
- `constraintScoringMode`
- `instanceFeatureNames`
- `abstractFeatureNames`

## Runtime Diagnostics

Constraint-aware hybrid payloads may include:
- `constraintAwareHybridUsed`
- `constraintScoringUsed`
- `reducedSolveApplied`
- `fixedCommitmentCount`
- `criticalConstraintCount`
- `deferredConstraintCount`
- `constraintReactivationCount`
- `stagedSolveRounds`
- `constraintConfidence`
- `reducedModelValidated`
- `reductionRejectedReason`
- `reducedSolveFallbackReason`

## Suggested Commands

Build a small dataset with constraint labels:

```bash
python scripts/build_power118_ml_dataset.py --num-samples 4 --seed 7 --time-limit-s 20 --output-dir backend_adapter/data/power118_dataset_constraint_v3
```

Train with constraint scoring enabled:

```bash
python scripts/train_power118_model.py --dataset-path backend_adapter/data/power118_dataset_constraint_v3/power118_ml_dataset.pkl --output-dir backend_adapter/data/power118_model --random-state 7
```

Compare warm-start and constraint-aware variants:

```bash
python scripts/eval_power118_modes.py --num-cases 1 --seed 7 --output-dir backend_adapter/data/power118_eval_constraint_v3 --time-limit-ms 20000 --require-exact-baseline
```

Check output consistency:

```bash
python scripts/check_power118_eval_consistency.py --records-path backend_adapter/data/power118_eval_constraint_v3/power118_eval_records.json --summary-path backend_adapter/data/power118_eval_constraint_v3/summary.json
```

Current evaluation labels:
- `hybrid_warm_start`
- `hybrid_constraint_aware_v2`
- `hybrid_constraint_aware_v3`

What to inspect for v3:
- `constraintScoringUsed`
- `criticalConstraintCount`
- `deferredConstraintCount`
- `constraintReactivationCount`
- `stagedSolveRounds`
- `reducedModelValidated`
- `reductionRejectedReason`
