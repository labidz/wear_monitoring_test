# Protocol v3: does the load-rise law anticipate end of life on a production-type machine?

Frozen: 2026-11-01 (Asia/Dhaka), before any model was fitted to the Haas data. Protocols v1 and v2 stay in force for
their datasets, and all of their results are reported.

## What has been seen

The Haas file was opened only to check its format, metadata and label ranges: 14 tools, cycles per tool, cutting
conditions per tool (tool type, ADOC, RDOC, holder length), workpiece hardness, and the cycles-to-failure label.
No sensor feature has been examined, plotted or correlated against the label. Tool lives range from 1 to 150 cycles,
and lives differ by up to 2x between tools cut under identical conditions. That is why the dataset was chosen: life
there cannot be predicted from time and conditions, which is the regime where v1 and v2 found that sensing should help.

## Data

Piecuch & Żabiński, Sci Data 12:650 (2025), doi:10.1038/s41597-025-04923-y. Kaggle copy grzegorzpiecuch/cnc-milling,
file FeatureAndMetadata_Milling.csv (968 cycles × 131 columns, CC BY 4.0).

- Machine and process: Haas VF-1, 42CrMo4 at 35.3–41.7 HRC, down-milling around the workpiece contour without
  retraction. Constant n = 3200 rpm, feed rate 640 mm/min, D = 10 mm, z = 4 (both catalogue tools are 4-flute end
  mills), so f_z = 0.05 mm and v_c = 100.5 m/min. a_p = ADOC (5 or 10 mm), a_e = RDOC (4.5 or 8 mm). Two cutters
  (Van Hoorn, type 1; PARA RS4, type 2). Holder length 80 or 160 mm.
- End of life is **breakage** of the cutter or of one tooth, not a flank-wear limit.
- Signals: per-cycle min/max/mean/std/skewness/kurtosis of 8 accelerometers and 12 current transformers
  (spindle L1–L3; X, Y and Z drives L1–L3), computed by the data owners over the whole cycle.

## Target and table

- Time = NumberOfCycle; pass = NumberOfCycle.
- Label: consumed life fraction y = 1 − CycleToFailureNormalized = (c − 1)/(L − 1), where L is the failure cycle
  (0 on the first cycle, 1 on the failure cycle). It is stored in the column VB, so all code runs unchanged.
  RMSE is reported in % of tool life.
- Primary load S = mean over L1–L3 of the per-cycle standard deviation of spindle current (AC RMS). No idle record
  exists, so no idle correction can be made; this is reported as a limitation.
- Reference: median S of the first three cycles of each tool (as in v2).
- Material key mat = workpiece–cutter pair (42CrMo4 with type 1 or type 2). P1 therefore fits one α per cutter type
  (5 constants), and every model sees the cutter type.
- h_m from f_z, a_e and D (physics.mean_chip_thickness).
- Further conditions known before cutting (workpiece hardness, holder length) enter every model that takes conditions
  (B1–B3, P2) as extra condition columns (prefix c_), so the clock baseline gets every piece of information a process
  planner has. P1 does not use them.
- Exclusion: tool 11 (P114) broke in its first cycle. The data owners call it a random outlier, and no reference can
  be formed. 13 tools remain, as the data owners recommend.

## Interpretation of the law for a breakage end of life (stated before fitting)

P1 is used unchanged: y = α_m · r · (h_m + h0) · a_p / (a_p + b_w). If a cutter breaks once its wear land reaches a
critical size, y ≈ VB/VB*, and α_m absorbs 1/VB*. If it breaks once the force reaches a critical value F*, the consumed
force margin is r·F0/F*, with F0 ∝ a_p(h_m + h0). That is the same form in the limit b_w ≫ a_p. Neither limit is imposed:
b_w is fitted within the v1 bounds.

## Hypothesis H3 and decision rule

H3: P3 (clock–load fusion) gives a lower RMSE of consumed life fraction than (a) the clock model B1 and (b) the
normalised-feature learner B3.

- Primary split: leave-one-tool-out (13 folds; the data owners' recommended protocol). The inner selection is
  leave-one-tool-out on the other 12.
- H3 is supported only if the 95 % cluster-bootstrap CI (B = 2000, over tools, paired) of the difference excludes zero
  for both P3 − B1 and P3 − B3. Otherwise it is reported as not supported.
- Secondary (reported, not part of the decision):
  - P1 − B1.
  - Leave-one-condition-out: the groups are the 7 combinations of cutter type × ADOC × RDOC × holder length, the
    inner loop is leave-one-condition-out, and all other settings are identical. This tests transfer to a condition
    never seen in training.
  - Coverage of 90 % group-conformal intervals.
  - Change decisions at τ = 0.9 of life: late means the tool entered its last 10 % of life before the change was
    triggered.
- Models: B0, B1, B2, B3, P1, P2, P3. B4 is not applicable, because only per-cycle statistics are available.
  Amendment v2-2 (fast P3 calibration) applies.

## Amendments

(none)
