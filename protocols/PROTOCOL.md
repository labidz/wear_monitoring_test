# Pre-registered analysis protocol

Frozen: 2026-11-01 (Asia/Dhaka), before any modelling on confirmatory datasets.
Changes after this date are logged in the "Amendments" section with date and reason, and are reported in the paper.

## 1. Question

Does a gain-invariant, mechanics-normalised spindle-load indicator, combined with a monotone learned mapping and
group-conformal calibration, estimate flank wear VB and support risk-bounded tool changes **under transfer** to unseen
cutting conditions, tools, machines and datasets better than (i) time-only, (ii) process (time + conditions) and
(iii) data-driven sensor models?

## 2. Datasets and their role

| Role | Dataset | Group unit for transfer | Load signal S |
|---|---|---|---|
| Development | NASA Ames milling (Agogino & Goebel 2007) | cutting setting (8; both replicates held out) | spindle AC current → torque component |
| Confirmatory | NUAA Ideahouse (Liu et al. 2021) | experiment / condition (W1–W9) | dynamometer torque; spindle power (sensorless variant) |
| Confirmatory | PHM 2010 (c1, c4, c6) | tool (3) | dynamometer in-plane force RMS |
| Confirmatory | Denkena, Klemme & Stiehl 2023 (3 machines) | machine (3) | NC spindle torque (sensorless); dynamometer force |
| Confirmatory (if obtained) | Nonastreda (2025) | tool (10) | dynamometer in-plane force RMS |
| Exploratory | cross-dataset zero-shot (train on all but one dataset) | dataset | as above |

NASA was used to develop the physics form (Section 3), so its results are labelled "development" in the paper.
Confirmatory datasets were inspected only for file format, signal ranges and label ranges before this freeze
(one audit printout of the downsampled PHM2010 bundle showed mean forces at five wear levels; no model was fitted).

NASA data-quality rules (fixed before modelling): case 6 (single run) excluded; runs whose steady-cut segment cannot be
detected are dropped; smcDC is used only for segmentation (it saturates); AE/vibration dropouts kept as recorded.

## 3. Physics indicator (form fixed; constants fitted on training groups only)

Per pass k of tool j: steady-cut mean load S_jk (torque-equivalent where available), idle-corrected.
Reference S_j0 = load of the first valid pass of tool j (unlabelled, fresh tool) at the same condition.

    r_jk   = S_jk / S_j0 - 1                                   (gain-invariant load-rise ratio)
    lambda = r_jk * (h_m + h_0) * a_p / (a_p + b_w)            (mechanics-normalised indicator)
    VB_phys = alpha_m * lambda

h_m mean uncut chip thickness from f_z, immersion and tool geometry; a_p axial depth (major edge length);
h_0 = K_te0/K_tc (edge-force length scale), b_w wear-land length outside a_p (nose/minor flank), alpha_m = K_tc/tau_w
(per work-material class). Derivation: linear edge-force milling model (Budak et al. 1996) with a flank-wear-land
contact force proportional to VB (Waldorf et al. 1998; Smithey et al. 2001).

Spindle current to torque (NASA): S = sqrt(I_cut^2 - I_idle^2) (quadrature of torque- and magnetising-current).

## 4. Models (all hyper-parameters chosen by inner grouped CV on the training pool only)

Baselines
- B0 time-only: isotonic regression of VB on cumulative cutting time.
- B1 process: Taylor-type log-linear wear law and gradient boosting on (time, conditions, material); better chosen in inner CV.
- B2 sensor-feature ML: gradient boosting, random forest and MLP on per-pass statistics + time + conditions.
- B3 per-tool normalised features (first-pass ratio for all features) + same learners (as in Özdamar 2026).
- B4 end-to-end deep learning on raw signal windows: 1D-CNN and CNN-GRU over the pass sequence, with conditions.

Proposed
- P1 physics-only: VB = alpha_m * lambda (3–4 constants).
- P2 hybrid (main): monotone neural network (non-negative weights on the lambda and time paths) taking
  lambda, r, time and conditions; ensemble of 5 seeds; online running-maximum along each tool's passes
  (irreversibility of wear).

Ablations of P2: without fresh-tool normalisation; without geometry term; without time; without monotonicity.

## 5. Evaluation

- Outer loop: leave-one-group-out per Section 2. Inner loop: leave-one-group-out on the training pool.
- Metrics: RMSE and MAE of VB (µm) pooled over held-out passes; per-group RMSE.
- Uncertainty of metrics: cluster bootstrap over held-out groups (B = 2000), paired for differences.
  Claim "better" only if the 95 % CI of the paired RMSE difference excludes zero.
- Neural models: 5 seeds; report mean and seed spread.
- Primary comparison: P2 versus the best of B1–B4 on each dataset (chosen on inner CV, not on test).

## 6. Calibrated intervals and tool-change decision

- Group-conformal CV+ with groups as exchangeable units; normalised score |y - yhat| / (c0 + c1 * yhat).
  Target coverage 90 %. Compared with ungrouped split conformal and with ensemble spread.
- Report empirical coverage (pooled and per held-out group) and mean interval width.
- Tool change when the upper bound reaches VB_lim. VB_lim: NASA 0.30 mm (ISO 8688-1 criterion),
  NUAA 0.20 mm, PHM2010 0.15 mm, Denkena 0.15 mm (dataset end-of-life). Metrics: late-change rate (true VB passed
  VB_lim before the change) and tool-life utilisation (change time / true crossing time, interpolated).
  Policies: conformal upper bound, point estimate, fixed tool life from training tools.

## 7. Reporting

All results reported whatever their direction, including where the proposed model does not win.
Compute cost (training time on a 2-core CPU, parameters, inference time per pass) reported for every model.

## Amendments

**Amendment 1 (2026-11-01, before any confirmatory modelling).** On the NASA development set the pre-registered
P2 (free partially monotone network on lambda, r and time) extrapolated outside the training range of r and time and
failed (pooled RMSE 1.05 mm versus 0.117 mm for P1). P2 is replaced by a physics-anchored network
VB = alpha_m * lambda * exp(kappa * tanh(g(z))), kappa = ln 2, z = conditions, material and normalised cutting time,
5-seed ensemble, running maximum along each tool's passes. The original design is still run on every dataset and
reported as "P2orig". Ablations now refer to the new P2: without time in z; without the mechanics term (r only);
without fresh-tool normalisation (raw load level); without the running maximum.

**Amendment 2 (2026-11-01, before any Denkena or PHM2010 data were read).** Data access: PHM2010 (Kaggle mirror
rabahba/phm-data-challenge-2010) and the Denkena dataset (Kaggle mirror humbleresearcher/luh-milling-dataset, a parquet
flattening of Mendeley 10.17632/zpxs87bjt8.3) are reduced on Kaggle with the frozen feature code; NUAA is not obtainable
(IEEE DataPort login) and is dropped. Denkena specifics, fixed from the data descriptor (Denkena et al. 2023) only:
(i) process: D = 8 mm, z = 4, v_c = 200 m/min, v_f = 1592 mm/min, a_e = 4 mm, a_p = 2 mm, cast iron 600-3/S;
(ii) primary load signal = NC spindle torque (sensorless), idle-corrected; dynamometer in-plane force is a secondary
variant; (iii) groups = machines (leave-one-machine-out, 3 folds); (iv) tool T8 lacks its first runs, so its reference
is its first recorded run; primary analysis keeps T8, a sensitivity analysis excludes it; (v) machine M2 torque is
aliased (control filter disabled) and is kept as recorded; (vi) VB labels between microscope measurements are linear
interpolations made by the data owners; all runs are used for training, metrics are reported on all runs and, as a
sensitivity analysis, only on runs at the breakpoints of each tool's piecewise-linear label trajectory (the microscope
measurement points); (vii) VB_lim = 0.15 mm.
PHM2010: label = max of the three flute wear values; load = mean in-plane force over the steady cut; VB_lim = 0.15 mm.

**Amendment 3 (2026-11-01, computational, before any confirmatory modelling).** Conformal calibration scores are the
inner leave-one-group-out predictions of the selected configuration, reused from model selection, for every model
(single-seed for neural models, which makes their intervals slightly conservative). This avoids refitting five-seed
ensembles inside every inner fold, which would take about three hours per dataset for B4 on a 2-core CPU. NASA results
are re-run with this rule so all datasets use the same procedure.

**Amendment 4 (2026-11-01, after reading PHM2010/Denkena load signals, before fitting any model to them).** The PHM2010
in-plane force rises from about 1.5 N to about 6 N over the first three cuts of every cutter (run-in), so a single
first-pass reference is noisy there; Denkena references are stable within about 2 %. The primary analysis keeps the
pre-registered reference (first valid pass). A sensitivity analysis, run on all datasets, uses a robust reference: the
median load of the first three valid passes. Both are reported. No other change.
