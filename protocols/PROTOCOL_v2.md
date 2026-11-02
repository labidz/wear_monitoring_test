# Protocol v2 — confirmatory test of clock–load fusion

Frozen: 2026-11-01 (Asia/Dhaka), before any model was fitted to NUAA data. Protocol v1 (PROTOCOL.md, amendments 1–4)
stays in force for NASA, PHM2010 and Denkena and all of its results are reported.

## Why a v2

Under protocol v1 the mechanics-normalised load law (P1/P2) beat all baselines across cutting settings on the NASA
development set (P2 114 µm vs 148 µm for the process model), but lost to the wear–time trend on the two
fixed-condition confirmatory sets (PHM2010: 145 vs 26 µm; Denkena: 21–43 vs 13–31 µm). Post-hoc inspection showed
that the errors of the clock model and of the load law are weakly or negatively correlated (NASA +0.27, Denkena −0.41),
so a pooled estimate could beat both. Because this was found after seeing those outcomes, NASA, PHM2010 and Denkena are
**development data** for v2; the v2 claim rests only on the confirmatory dataset below.

## Hypothesis H2

A log-linear pool of the wear-clock model (B1) and the mechanics law (P1), with the pooling weight chosen by inner
leave-one-group-out, gives lower RMSE than (a) the clock model B1 and (b) the strongest sensor baseline of development,
per-tool-normalised feature learning B3, when a whole cutting condition is held out.

## Confirmatory data

NUAA Ideahouse tool-wear dataset (Ti6Al4V, orthogonal design W1–W9: nine combinations of feed per tooth, spindle speed
and axial depth, one tool per combination). Primary source: the original IEEE DataPort files. If they cannot be
obtained, the downsampled copy in the Uniwear repository (katulu-io) is used and this is reported as a limitation.
Only file format, signal ranges and label ranges have been inspected; no wear-versus-signal relation has been examined.

- Groups: experiments W1–W9 (leave-one-condition-out, 9 folds). Inner loop: leave-one-group-out on the other 8.
- Label: VB = maximum flank wear over the cutting edges (mm). VB_lim = 0.20 mm.
- Primary load signal: spindle torque from the rotating dynamometer, mean over the steady cut, idle-corrected.
  Sensorless secondary: spindle power, idle-corrected.
- Reference: median load of the first three valid passes of each tool (adopted from development, amendment 4).
- Tool geometry and conditions for h_m are taken from the dataset documentation before modelling.

## Models

All v1 baselines (B0–B4), P1, P2, and the new P3 (code/models.py, ClockLoadFusion):
log(VB + ε) = w log(VB_B1 + ε) + (1 − w) log(VB_P1 + ε), ε = 0.01 mm, w on a 0.05 grid chosen by inner
leave-one-group-out, followed by a running maximum along each tool's passes. Intervals: group-conformal as in v1.

## Decision rule

Pooled RMSE over held-out passes; 95 % cluster-bootstrap CI over the 9 held-out conditions (B = 2000), paired.
H2 is supported only if the CI of the paired difference excludes zero for **both** comparisons (P3 − B1 and P3 − B3).
Otherwise H2 is reported as not supported. Secondary: coverage, late-change rate and tool-life utilisation.
P3 is also run on the development datasets and reported as development results.

## Amendments

**Amendment v2-1 (2026-11-01, before any model was fitted to NUAA or Vicomtech data).**
(a) The original NUAA files require an IEEE DataPort subscription and could not be obtained; the pre-declared fallback
(Uniwear downsampled copy, ~15 Hz, smoothing undocumented) is used and reported as a limitation. NUAA remains the
**primary** confirmatory test of H2 because it is the only one with varying conditions.
(b) A second, untouched confirmatory dataset is added: Vicomtech/Mondragon turning data (GitHub, CC BY-SA 4.0;
Badiola et al., Int J Adv Manuf Technol 2023): 19NiMoCr6, uncoated P25 insert, v_c = 200 m/min, f = 0.1 mm/rev,
a_p = 2 mm (fixed conditions), 13 tools, 1-s segments with 5 statistics per signal, VB measured 3–9 times per tool and
linearly interpolated by the data owners. Groups: tools (leave-one-tool-out). VB_lim = 0.25 mm (ISO 3685, as in the
source). Primary load: mean tangential cutting force F_c (dynamometer); sensorless secondary: mean spindle motor torque
TV2.S (no idle record available). Time = segment index (s). B4 is not applicable (no raw signals). Tools whose first
segment is already worn (VB > 0) keep their first segments as reference; a sensitivity analysis excludes them.
(c) NUAA reduction rules: pass = source file (experiment_csv_n); steady segment = samples with spindle power above
idle + 0.5 (p90 − idle), idle = 5th percentile of the pass; primary load = mean torsion over the segment; sensorless
secondary = mean spindle power minus idle; VB = maximum of the four blade wear values at the pass midpoint; time =
timestamp at the pass midpoint; conditions from the bundle (f_z, n, a_p); D, z and a_e are not documented in the bundle
and are held constant (they only rescale h_m, which the fitted constants absorb within a dataset). CNN windows: six
channels resampled to 256 points per pass.
(d) Decision rule unchanged and applied to each dataset separately. H2 is called supported if it holds on NUAA;
Vicomtech is reported as a secondary confirmation at fixed conditions.

**Amendment v2-2 (2026-11-01, computational, before any P3 result on Vicomtech was seen).** For P3 the conformal
calibration scores are its own inner out-of-group pooled predictions (the pooling weight is chosen on the same inner
predictions, a mild optimism that can only narrow its intervals). Refitting the nested model for every inner group
took about 15 min per outer fold on Vicomtech. Point predictions are unaffected. Applied to all datasets for P3.

**Amendment v2-3 (2026-11-01, computational, before any B2/B3 result on Vicomtech was seen).** B3 on Vicomtech stopped
with an error: skewness and kurtosis of the X-axis velocity (CV3.X) are undefined in 571 one-second segments where that
signal is constant. They are set to 0, the rule already used for NUAA (amendment v2-1 c). No other column is affected,
and the load signals used by P1–P3 are unchanged.

**Erratum (2026-11-01, bibliographic only).** The Vicomtech/Mondragon source paper is Fernández de Barrena et al.
(Int J Adv Manuf Technol 125:4027–45, 2023, doi:10.1007/s00170-023-10811-9); Badiola is its fourth author.
