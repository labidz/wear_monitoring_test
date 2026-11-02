PHM2010: P2 and A_noTime coincide (max diff 6e-9 mm) because the inner loop chose wd=0.1 in all folds; the correction reduced to a per-fold constant factor (1.54, 1.00, 1.00), so the time input had no effect. Not a bug.

## Haas (v3) — post-hoc, explanatory (after the pre-registered result was seen)
- Pre-registered P1 fit: alpha(type 1) hit the lower bound (~0), alpha(type 2) = 60, b_w at its upper bound 20 mm;
  P1 predicts ~0 for the 9 type-1 tools -> LOTO RMSE 55 % of life.
- Not a loss-scale artefact: soft-L1 f_scale 0.05 / 0.25 / 1.0 -> 55.2 / 55.0 / 53.9 % (refit LOTO, post hoc).
- Spindle-current rise r does grow with consumed life on most tools: per-tool Spearman(r, life) median 0.55
  (range 0.21-0.88), but the median r in the last 10 % of life ranges from 0.04 to 0.92 between tools, and two tools
  (H9, H10) stay below their reference all life (median r -0.44 on H9's first workpiece at constant hardness, i.e. an
  unrepresentative first-cycle reference, not a hardness change). Breakage does not occur at a fixed load rise.

## Compute note
From 22:55 on 2026-11-01 the queues run single-threaded (TORCH_THREADS=1, OMP/OPENBLAS=1) after thread oversubscription slowed neural fits ~15x. Folds completed earlier used 2 threads. Thread count can change floating-point summation order in the MLP and neural fits at round-off level only; every fold is computed entirely under one setting.
