# When does sensing beat the tool-life clock?

Code, pre-registered protocols and held-out predictions for:

> R. Amin, M. R. Rana, S. Aktar. *When does sensing beat the tool-life clock? A pre-registered test of wear monitoring
> on six datasets.* Submitted to CIRP Annals.

The study compares a wear "clock" (cutting time + planned conditions) with a four-constant mechanics law, a
physics-anchored network, a clock–load fusion and data-driven monitors (feature learning, normalised feature learning,
1D-CNN) on six public datasets, using nested leave-one-group-out evaluation, paired cluster-bootstrap intervals,
group-conformal bounds and tool-change decisions.

## Repository layout

| Path | Content |
|---|---|
| `protocols/` | The three pre-registered protocols with all amendments, and `git_log.txt`, the time-stamped history showing that each protocol and amendment precedes the results it governs |
| `code/` | Analysis code (see below) |
| `code/kaggle_reduction/` | Scripts run on Kaggle to reduce the large raw files of PHM 2010 and Hannover to per-pass statistics (`kaggle_main.py` was run; `probe*.py` are the format audits) |
| `results/<DATASET>/` | Held-out predictions (`<model>_preds.csv`, with 90 % group-conformal and naive intervals), costs and selected hyper-parameters (`<model>_cost.csv`), inner-CV scores (`<model>_inner.csv`) and metrics (`summary.csv`) |
| `results/contrasts.csv`, `sensitivity.csv`, `p3_weights.csv` | Pre-registered contrasts, sensitivity analyses, fusion weights per fold |
| `results/NOTES.md` | Post-hoc diagnostics, kept separate from the pre-registered analyses |
| `work/` | Per-pass tables are written here by the adapters (not distributed) |

## Data

No data are redistributed. Obtain each dataset from its source and reduce it with the matching adapter:

| Key | Dataset | Source | Adapter |
|---|---|---|---|
| NASA | NASA Ames milling (Agogino & Goebel 2007) | https://data.nasa.gov/dataset/milling-wear | `adapt_nasa.py <mill.mat> <runs.csv> <out.csv> <out.npz>` |
| PHM2010 | PHM Society 2010 data challenge (Li et al. 2009) | PHM Society / Kaggle mirror | `kaggle_reduction/kaggle_main.py`, then `adapt_reduced.py PHM2010 <dir>` |
| DENKENA | Hannover milling, three machines (Denkena et al. 2023) | https://doi.org/10.17632/zpxs87bjt8 | `kaggle_reduction/kaggle_main.py`, then `adapt_reduced.py DENKENA <dir> [force]` |
| NUAA | NUAA Ideahouse (Li et al. 2021), Uniwear bundle | https://doi.org/10.21227/3aa1-5e83, https://github.com/katulu-io/uniwear-dataset | `adapt_v2.py NUAA <nuaa_orthogonal_bundle_high_resolution.csv>` |
| VICOMTECH | Mondragon turning (Fernández de Barrena et al. 2023) | dataset linked from https://doi.org/10.1007/s00170-023-10811-9 (CC BY-SA 4.0) | `adapt_v2.py VICOMTECH <Dataset_Machine_tool_wear.csv>` |
| HAAS | Haas VF-1 run-to-failure (Piecuch & Żabiński 2025) | https://doi.org/10.6084/m9.figshare.28589216 (Kaggle: grzegorzpiecuch/cnc-milling) | `adapt_haas.py <FeatureAndMetadata_Milling.csv>` |

Suffixes: `_SENSORLESS` (sensorless load), `_R3` (three-pass reference), `_F` (Hannover dynamometer force),
`_LOCO` (Haas, leave one condition out) are written by the same adapters.

## Reproducing the results

```bash
pip install -r requirements.txt
cd code
python evaluate.py <DATASET> B0 B1 B2 B3 B4 P1 P2 P3     # nested leave-one-group-out; resumable per fold
python analyse.py <DATASET>                               # metrics, bootstrap CIs, coverage, decisions
python inner_scores.py <DATASET> B1 B2 B3 B4              # inner-CV scores for the best-baseline selection
python primary.py NASA PHM2010 DENKENA NUAA VICOMTECH HAAS
python sensitivity.py
python weights.py NASA PHM2010 DENKENA NUAA VICOMTECH HAAS
python tables_tex.py; python tables_decision.py
python figs2.py clock; python figs2.py forest NASA PHM2010 DENKENA NUAA VICOMTECH HAAS
```

Model keys: B0 time only; B1 clock (time + conditions); B2 signal features; B3 per-tool normalised features;
B4 1D-CNN; P1 four-constant load law; P2 physics-anchored network; P3 clock–load fusion; `A_*` ablations of P2;
`P2orig` the originally registered network. Runs were made on a two-core CPU with Python 3.13; setting
`TORCH_THREADS=1 OMP_NUM_THREADS=1` avoids thread oversubscription when several jobs share a CPU.

## Licence

To be chosen by the authors before the repository is made public.
