"""
Provenance:
This benchmarking script is adapted for use in the CAPRA repository from the
scPerturBench benchmarking framework.

Source:
bm2-lab/scPerturBench. GitHub repository.
URL: https://github.com/bm2-lab/scPerturBench.git
Accessed: 2026-04-28.

Local modifications in this repository mainly concern path resolution,
environment setup, and benchmark integration.
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
EMBEDDING_PATH = REPO_ROOT / "data" / "gene_embedding" / "GenePT_v2_raw" / "GenePT_gene_embedding_ada_text.pickle"
GENEPERT_REPO = Path(os.environ.get("GENEPERT_REPO", str(REPO_ROOT.parent / "GenePert")))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if GENEPERT_REPO.is_dir() and str(GENEPERT_REPO) not in sys.path:
    sys.path.insert(0, str(GENEPERT_REPO))

from myUtil1 import *
import torch
from collections import OrderedDict
from itertools import chain
import importlib
import matplotlib.pyplot as plt
import pickle, sklearn, umap                              
# Reload the module
import utils # type: ignore
import GenePertExperiment  #type: ignore
importlib.reload(utils)
# Reload the module
importlib.reload(GenePertExperiment)
from utils import get_best_overall_mse_corr, run_experiments_with_embeddings, plot_mse_corr_comparison, compare_embedding_correlations #type: ignore

import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import KFold
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity
import json

def get_gears_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "GEARS"


def get_genepert_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "GenePert"


def get_gears_processed_adata_path(dataset_name):
    return get_gears_dir(dataset_name) / "data" / "train" / "perturb_processed.h5ad"


def getcondition(DataSet, seed = 1):
    filein = get_gears_dir(DataSet) / "data" / "train" / "splits" / f"train_simulation_{seed}_0.8.pkl"
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
    train_conditions = mydict['train']
    test_conditions = mydict['test']
    return train_conditions, test_conditions

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def populate_dicts(adata_subset, mean_dict):
    for condition in adata_subset.obs['condition'].unique():
        condition_mask = adata_subset.obs['condition'] == condition
        condition_data = adata_subset[condition_mask].X
        clean_cond = clean_condition(condition)
        mean_dict[clean_cond] = np.mean(condition_data, axis=0)

def doLinearModel(DataSet, seed = 1):
    dirName = get_genepert_dir(DataSet)
    if not dirName.is_dir():
        dirName.mkdir(parents=True)
    os.chdir(dirName)

    dataset_path = get_gears_processed_adata_path(DataSet)
    experiment = GenePertExperiment.GenePertExperiment(embeddings=None)
    experiment.load_dataset(dataset_path)
    with open(EMBEDDING_PATH, "rb") as fp:
        embeddings = pickle.load(fp)
    experiment.embeddings = embeddings
    train_conditions, test_conditions = getcondition(DataSet, seed)
    embedding_size = len(next(iter(experiment.embeddings.values())))
    X_train, y_train, X_test, y_test = [], [], [], []
    train_mask = experiment.adata.obs["condition"].isin(train_conditions)
    test_mask = experiment.adata.obs["condition"].isin(test_conditions)
    adata_train = experiment.adata[train_mask]
    adata_test = experiment.adata[test_mask]
    mean_dict_train, mean_dict_test = {}, {}
    populate_dicts(adata_train, mean_dict_train)
    populate_dicts(adata_test, mean_dict_test)
    train_gene_name_X_map = experiment.populate_X_y(mean_dict_train, X_train, y_train, embedding_size)
    test_gene_name_X_map = experiment.populate_X_y(mean_dict_test, X_test, y_test, embedding_size)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    ridge_model = Ridge(alpha=1,  random_state=42)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    result = pd.DataFrame(y_pred, columns=experiment.adata.var_names, index=mean_dict_test.keys())
    dirOut = dirName / f"savedModels{seed}"
    if not dirOut.is_dir():
        dirOut.mkdir(parents=True)
    result.to_csv(dirOut / "pred.tsv", sep='\t')


def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


### 根据预测的生成表达量
def generateH5ad(DataSet, seed = 1):
    dirName = get_genepert_dir(DataSet)
    os.chdir(dirName)
    filein = dirName / f"savedModels{seed}" / "pred.tsv"
    exp = pd.read_csv(filein, sep='\t', index_col=0)
    filein = get_gears_dir(DataSet) / f"savedModels{seed}" / "result.h5ad"
    adata = sc.read_h5ad(filein)
    expGene = np.intersect1d(adata.var_names, exp.columns)
    pertGenes = np.intersect1d(adata.obs['perturbation'].unique(), exp.index)
    adata = adata[:, expGene].copy()
    exp = exp.loc[:, expGene]

    control_exp = adata[adata.obs['perturbation'] == 'control'].to_df()
    control_std = np.asarray(np.std(control_exp, axis=0), dtype=np.float32)
    control_std[np.isnan(control_std)] = 0

    imputed_mask = adata.obs['Expcategory'].astype(str).to_numpy() == 'imputed'
    row_order = np.arange(adata.n_obs, dtype=np.int64)

    adata_imputed = adata[imputed_mask].copy()
    adata_other = adata[~imputed_mask].copy()

    if sparse.issparse(adata_imputed.X):
        imputed_matrix = adata_imputed.X.toarray().astype(np.float32, copy=False)
    else:
        imputed_matrix = np.asarray(adata_imputed.X, dtype=np.float32).copy()

    imputed_perts = adata_imputed.obs['perturbation'].astype(str).to_numpy()
    for pertGene in pertGenes:
        row_mask = imputed_perts == pertGene
        cellNum = int(row_mask.sum())
        if cellNum == 0:
            continue
        means = exp.loc[pertGene].to_numpy(dtype=np.float32, copy=False)
        imputed_matrix[row_mask] = generateExp(cellNum, means, control_std).astype(np.float32, copy=False)

    adata_imputed.X = sparse.csr_matrix(imputed_matrix)
    adata_imputed.obs['_row_order'] = row_order[imputed_mask]
    adata_other.obs['_row_order'] = row_order[~imputed_mask]

    adata = ad.concat([adata_imputed, adata_other], merge='same')
    adata = adata[np.argsort(adata.obs['_row_order'].to_numpy())].copy()
    adata.obs.drop(columns=['_row_order'], inplace=True)
    adata.write(dirName / f"savedModels{seed}" / "result.h5ad")



### conda activate gears
seeds = [1, 2, 3, 4, 5]

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]


if __name__ == '__main__':
    print ('hello, world')
    for myDataSet in ["Schmidt"]:
        print (myDataSet)
        for seed in tqdm(seeds):
            doLinearModel(myDataSet, seed)
            generateH5ad(myDataSet, seed)
