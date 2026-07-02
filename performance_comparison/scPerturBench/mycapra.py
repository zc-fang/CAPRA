"""
Provenance:
This benchmarking wrapper is implemented in the CAPRA repository within a
workflow adapted from the scPerturBench benchmarking framework.

Source benchmark framework:
bm2-lab/scPerturBench. GitHub repository.
URL: https://github.com/bm2-lab/scPerturBench.git
Accessed: 2026-04-28.

Local modifications in this repository mainly concern CAPRA-specific method
integration, path resolution, and benchmark execution.
"""

import gc
import os
import random
import sys
import time
import warnings
from itertools import chain
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPRA_SOURCE_ROOT = REPO_ROOT / "capra"
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
CAPRA_OUTPUT_ROOT = Path(os.environ.get("CAPRA_OUTPUT_ROOT", DATASETS_ROOT)).expanduser().resolve()
CAPRA_WORKSPACE_ROOT = Path(os.environ.get("CAPRA_WORKSPACE_ROOT", REPO_ROOT / "tmp")).expanduser().resolve()
EMBEDDING_PATH = REPO_ROOT / "data" / "gene_embedding" / "processed" / "genept_embeddings.pkl"
CAPRA_GLOBAL_SEED = 24

if str(CAPRA_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(CAPRA_SOURCE_ROOT))

import pickle
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from frame import CAPRA, CAPRAData
from tqdm import tqdm


def clean_condition(condition):
    return condition.replace("+ctrl", "").replace("ctrl+", "").strip()


def condition_sort(x):
    return "+".join(sorted(x.split("+")))


def get_capra_dir(dataset_name):
    return CAPRA_OUTPUT_ROOT / dataset_name / "hvg5000" / "capra"


def get_gears_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "GEARS"


def _as_float32_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _result_var_from_source(adata):
    return pd.DataFrame(index=adata.var_names.copy())


def trainModel(DataSet, seed):
    DataSet = str(DataSet)
    split_seed = int(seed)
    dirName = get_capra_dir(DataSet)
    trainDir = get_gears_dir(DataSet) / "data" / "train"

    if not dirName.is_dir():
        dirName.mkdir(parents=True)
    os.chdir(dirName)

    adata_path = trainDir / "perturb_processed.h5ad"
    filein = trainDir / "splits" / f"train_simulation_{split_seed}_0.8.pkl"

    if not os.path.exists(adata_path):
        print(f"Dataset file not found: {adata_path}")
        return
    if not os.path.exists(filein):
        print(f"Split file not found: {filein}")
        return

    adata = sc.read_h5ad(adata_path)
    adata.obs["condition"] = adata.obs["condition"].astype(str).apply(lambda x: condition_sort(x)).astype("category")
    adata.obs["perturbation"] = (
        adata.obs["perturbation"].astype(str).apply(lambda x: clean_condition(condition_sort(x))).astype("category")
    )
    adata.uns = {}
    if "condition_name" in adata.obs.columns:
        adata.obs.drop("condition_name", axis=1, inplace=True)

    with open(EMBEDDING_PATH, "rb") as f:
        embd = pd.DataFrame(pickle.load(f)).T
    embd.index = embd.index.astype(str)
    ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=["ctrl"])
    embd_all = pd.concat([ctrl_row, embd]).astype(np.float32)

    with open(filein, "rb") as fin:
        splits = pickle.load(fin)
    splits = {key: [condition_sort(str(item)) for item in values] for key, values in splits.items()}

    embd_index = set(embd_all.index.astype(str))

    def perturb_genes(condition):
        condition = clean_condition(condition_sort(str(condition)))
        return [gene for gene in condition.split("+") if gene not in {"", "ctrl", "control"}]

    dropped = {key: [] for key in ("train", "val", "test")}
    for split_name in ("train", "val", "test"):
        kept = []
        for condition in splits[split_name]:
            if all(gene in embd_index for gene in perturb_genes(condition)):
                kept.append(condition)
            else:
                dropped[split_name].append(condition)
        splits[split_name] = kept

    dropped_total = sorted({condition for values in dropped.values() for condition in values})
    if dropped_total:
        warnings.warn(
            f"CAPRA skipped {len(dropped_total)} split conditions without GenePT embeddings",
            RuntimeWarning,
        )
        for split_name in ("train", "val", "test"):
            if dropped[split_name]:
                warnings.warn(
                    f"CAPRA skipped {split_name} conditions: {sorted(set(dropped[split_name]))}",
                    RuntimeWarning,
                )

    valid_conditions = set(splits["train"] + splits["val"] + splits["test"])
    keep_mask = (adata.obs["perturbation"].astype(str) == "control") | adata.obs["condition"].astype(str).isin(valid_conditions)
    adata = adata[keep_mask].copy()

    capra_data = CAPRAData(
        adata=adata,
        embedding_table=embd_all,
        condition_key="condition",
        perturbation_key="perturbation",
    )
    capra_data.harmonize_perturbation_metadata()
    capra_data.register_evaluation_partitions(split_dict=splits)
    capra_data.estimate_trainval_deg_reference(method="t-test")
    capra_data.build_control_relative_training_state(
        topk_deg=100,
        knn_topk=5,
        knn_temperature=12.0,
        deg_method="t-test",
    )

    workspace_root = CAPRA_WORKSPACE_ROOT / f"capra_{DataSet}_seed{split_seed}"
    capra_model = CAPRA(capra_data)
    capra_model.fit_capra_response_operator(
        output_dir=str(workspace_root / "src" / "results"),
        run_name=f"capra_response_operator_seed{split_seed}",
        n_epochs=80,
        min_epochs=20,
        patience=10,
        num_workers=8,
        pin_memory=True,
        seed=24,
        batch_size=192,
        learning_rate=6e-4,
        weight_decay=1e-4,
    )

    pred_dict = capra_model.generate_counterfactual_profiles(
        pert_list=splits["test"],
        n_pred=500,
    )

    adata_list = []
    for pertGene in pred_dict:
        tmp_adata = ad.AnnData(pred_dict[pertGene])
        tmp_adata.obs['perturbation'] = clean_condition(pertGene)
        adata_list.append(tmp_adata)
    adata_pred = ad.concat(adata_list)
    adata_pred.obs['Expcategory'] = 'imputed'
    adata_pred.var = adata.var

    adata_control = adata[adata.obs['perturbation'] == 'control']
    adata_control.obs['Expcategory'] = 'control'
    adata_truth = adata[adata.obs['perturbation'].isin(list(adata_pred.obs['perturbation'].unique()))]
    adata_truth.obs['Expcategory'] = 'stimulated'
    result = ad.concat([adata_pred, adata_control, adata_truth])
    dirOut = 'savedModels{}'.format(seed)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.write_h5ad('{}/result.h5ad'.format(dirOut))


### conda activate perturb
seeds = [1, 2, 3, 4, 5]

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # for myDataSet in tqdm(chain(['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential"])):
    # for myDataSet in tqdm(chain(["Replogle_K562essential"],CombPertDataSets)):
    for myDataSet in tqdm(chain(SinglePertDataSets, CombPertDataSets)):
        for seed in tqdm(seeds):
            trainModel(myDataSet, seed)
