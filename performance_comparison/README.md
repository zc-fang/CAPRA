# Benchmark integration

`scPerturBench/` adapts [bm2-lab/scPerturBench](https://github.com/bm2-lab/scPerturBench) for CAPRA (source accessed 2026-04-28; exact upstream commit unrecorded). The wrappers use explicit input, workspace and result paths; `calPerformance_genetic.py` follows the local no-argument evaluation script. This directory carries the applicable [GNU GPL v3 license](LICENSE). Cite Wei et al., *Nature Methods* **23**, 451–464 (2026), [DOI:10.1038/s41592-025-02980-0](https://doi.org/10.1038/s41592-025-02980-0).

## Data and output layout

Set `CAPRA_DATA_ROOT` to the input root containing `datasets/` and `gene_embedding/` (default: repository `data/`). For each dataset, the wrappers use `datasets/DATASET/filter_hvg5000_logNor.h5ad` or `datasets/DATASET/hvg5000/GEARS/data/train/perturb_processed.h5ad`, the released `splits/train_simulation_SEED_0.8.pkl`, and `datasets/DATASET/DEG_hvg5000.pkl` for scoring. CAPRA and scouter use `gene_embedding/processed/genept_embeddings.pkl`; GenePert uses its raw GenePT asset. scGPT requires pretrained weights.

`CAPRA_RESULTS_ROOT` defaults to `tmp/benchmark_results` and stores `DATASET/hvg5000/METHOD/savedModelsSEED/result.h5ad`. `CAPRA_WORKSPACE_ROOT` defaults to `tmp/benchmark_runs` for caches. Both directories must be outside the input tree. `CAPRA_OUTPUT_ROOT` is an alias for the results root. The GEARS helper copies released inputs into its working directory before execution. The R linear model uses the `plot` conda environment by default (`CAPRA_R_ENV` overrides it).

## Run CAPRA

The CAPRA wrapper reads the released split selected by seed 1–5, uses training seed 24 and the study's 80-epoch / 500-predicted-cell settings. Run one dataset and split from the repository root:

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('performance_comparison/scPerturBench').resolve()))
from mycapra import trainModel
trainModel('Norman', 1)
PY
```

The GEARS, scouter, GenePert and scGPT wrappers use their respective upstream environments and assets.

## Calculate performance

From `performance_comparison/`, run the local evaluation script in a Python environment with `pertpy`:

```bash
python3 calPerformance_genetic.py
```

The script reads predictions and DEG tables from `data/datasets/`. Its dataset, method, seed and metric lists are set at the end of the script; `ff_calPerfor` specifies the output TSV path. The default method is `linearModel`.

The existing `results/genetic_results.tsv` has 268,281 rows covering 13 datasets, five seeds and six method labels. `linearModel` is a supplementary comparison. The table is included as released; use the supplied benchmark splits and DEG files when generating new scores.
