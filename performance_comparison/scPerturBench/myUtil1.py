"""
Provenance:
This utility module is adapted for use in the CAPRA repository from the
scPerturBench benchmarking framework.

Source:
bm2-lab/scPerturBench. GitHub repository.
URL: https://github.com/bm2-lab/scPerturBench.git
Accessed: 2026-04-28.

Local modifications in this repository mainly concern path resolution and
benchmark integration.

License: GNU GPL v3; see ../LICENSE. Original applicable notices remain in force.
Modified: 2026-09-30. Keep source datasets read-only and route generated files
to separate run/result directories. The original source commit was not recorded.
"""

import subprocess, os, sys, re, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from multiprocessing import Pool
import multiprocessing
from sklearn.utils import shuffle
from tqdm import tqdm
from scipy.sparse import csr_matrix, issparse, spmatrix
from anndata import AnnData
from scipy import sparse
import anndata as ad
import scanpy as sc
import logging
from collections import defaultdict, OrderedDict
from itertools import chain
import warnings
warnings.filterwarnings('ignore')
import pickle
import shutil

SOURCE_REPO_ROOT = Path(os.environ.get("CAPRA_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
REPO_ROOT = SOURCE_REPO_ROOT


def resolve_data_root():
    """Return the read-only data root (contains datasets/ and gene_embedding/)."""
    return Path(os.environ.get('CAPRA_DATA_ROOT', SOURCE_REPO_ROOT / 'data')).expanduser().resolve()


def _writable_root(env_name, default):
    root = Path(os.environ.get(env_name, default)).expanduser().resolve()
    data_root = resolve_data_root()
    if root == data_root or data_root in root.parents:
        raise ValueError(f'{env_name} must be outside the read-only data tree: {root}')
    return root


def resolve_results_root():
    # Retain the older CAPRA_OUTPUT_ROOT alias for existing CAPRA runs.
    return _writable_root('CAPRA_RESULTS_ROOT', os.environ.get('CAPRA_OUTPUT_ROOT', SOURCE_REPO_ROOT / 'tmp/benchmark_results'))


def resolve_workspace_root():
    return _writable_root('CAPRA_WORKSPACE_ROOT', SOURCE_REPO_ROOT / 'tmp/benchmark_runs')


WORKSPACE_ROOT = resolve_workspace_root()
DATASETS_ROOT = resolve_data_root() / 'datasets'
RESULTS_ROOT = resolve_results_root()


def get_gears_input_dir(dataset_name):
    return DATASETS_ROOT / str(dataset_name) / 'hvg5000/GEARS'


def get_gears_work_dir(dataset_name):
    return RESULTS_ROOT / str(dataset_name) / 'hvg5000/GEARS'


def get_gears_result_path(dataset_name, seed):
    """Read a generated GEARS reference, or an existing released result."""
    relative = Path(f'savedModels{int(seed)}') / 'result.h5ad'
    generated = get_gears_work_dir(dataset_name) / relative
    if generated.is_file():
        return generated
    released = get_gears_input_dir(dataset_name) / relative
    if released.is_file():
        return released
    raise FileNotFoundError(f'GEARS reference result not found: {generated} or {released}')


def prepare_gears_workspace(dataset_name, seed):
    """Copy released GEARS inputs before a library writes caches or split files."""
    source = get_gears_input_dir(dataset_name) / 'data'
    processed = source / 'train/perturb_processed.h5ad'
    split_name = f'train_simulation_{int(seed)}_0.8.pkl'
    source_split = source / 'train/splits' / split_name
    for required in (processed, source_split):
        if not required.is_file():
            raise FileNotFoundError(f'Released benchmark input required: {required}')
    destination = get_gears_work_dir(dataset_name)
    working_data = destination / 'data'
    if not working_data.exists():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, working_data)
    elif not (working_data / 'train').exists():
        # Restore the released inputs after an explicit redo cleared its cache.
        shutil.copytree(source / 'train', working_data / 'train')
    elif not (working_data / 'train/perturb_processed.h5ad').is_file():
        raise RuntimeError(f'Incomplete benchmark working copy: {working_data}')
    # A library may have removed its cached split; restore the released split,
    # rather than silently generating a different evaluation partition.
    target_split = working_data / 'train/splits' / split_name
    target_split.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_split, target_split)
    return destination

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def subSample(adata, n_samples):
    if adata.shape[0] <= n_samples:
        return adata
    else:
        sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
        adata_sampled = adata[sampled_indices, :]
        return adata_sampled


def preData(adata, domaxNumsPerturb=0, domaxNumsControl=0, minNums = 50, min_cells= 10):
    adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata = adata[~adata.obs.index.duplicated()]
    adata = adata[adata.obs["perturbation"] != "None"]
    filterNoneNums = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes= 200)
    sc.pp.filter_genes(adata, min_cells= min_cells)
    filterCells = adata.shape[0]

    if np.any([True if i.startswith('mt-') else False for i in adata.var_names]):
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
    else:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if sum(adata.obs['pct_counts_mt'] < 10) / adata.shape[0] <=0.5:
        adata = adata[adata.obs.pct_counts_mt < 15, :]
    else:
        adata = adata[adata.obs.pct_counts_mt < 10, :]
    filterMT = adata.shape[0]
    tmp = adata.obs['perturbation'].value_counts()
    tmp_bool = tmp >= minNums
    genes = list(tmp[tmp_bool].index)
    if 'control' not in genes: genes += ['control']
    adata = adata[adata.obs['perturbation'].isin(genes), :]
    filterMinNums = adata.shape[0]

    if domaxNumsPerturb:
        adata1 = adata[adata.obs['perturbation'] == 'control']
        perturbations = adata.obs['perturbation'].unique()
        perturbations = [i for i in perturbations if i != 'control']
        adata_list = []
        for perturbation in perturbations:
            adata_tmp = adata[adata.obs['perturbation'] == perturbation]
            adata_tmp = subSample(adata_tmp, domaxNumsPerturb)
            adata_list.append(adata_tmp)
        adata2 = ad.concat(adata_list)
        adata = ad.concat([adata1, adata2])
        adata.var = adata1.var.copy()
    if domaxNumsControl:
        adata1 = adata[adata.obs['perturbation'] == 'control']
        adata2 = adata[adata.obs['perturbation'] != 'control']
        adata1 = subSample(adata1, domaxNumsControl)
        adata = ad.concat([adata1, adata2])
        adata.var = adata1.var.copy()

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4) 
    sc.pp.log1p(adata)
    adata.layers['logNor'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)
    adata.var['highly_variable_5000'] = adata.var['highly_variable']
    adata = adata[adata.obs.sort_values(by='perturbation').index,:]
    return filterNoneNums, filterCells, filterMT, filterMinNums, adata

def calDEG(DataSet='Adamson', condition_column='perturbation', control_tag='control', adata=None, fileout=None, return_dict=False):
    if fileout is not None:
        fileout = Path(fileout).expanduser().resolve()
        if fileout == resolve_data_root() or resolve_data_root() in fileout.parents:
            raise ValueError('Computed DEG outputs must be outside the read-only data tree')
    if adata is None:
        dataset_dir = DATASETS_ROOT / DataSet
        filein = dataset_dir / 'filter_hvg5000_logNor.h5ad'
        if fileout is None:
            fileout = WORKSPACE_ROOT / 'computed_deg' / DataSet / 'DEG_hvg5000.pkl'
        adata = sc.read_h5ad(filein)
    else:
        adata = adata.copy()
    adata.uns['log1p'] = {}
    adata.uns['log1p']['base'] = None
    mydict = defaultdict(dict)
    perturbations = adata.obs[condition_column].unique()
    perturbations = [i for i in perturbations if i != control_tag]
    sc.tl.rank_genes_groups(adata, condition_column, groups=perturbations, reference= control_tag, method= 't-test')
    result = adata.uns['rank_genes_groups']
    for perturbation in perturbations:
        final_result = pd.DataFrame({key: result[key][perturbation] for key in ['names', 'pvals_adj', 'logfoldchanges', 'scores']})
        tmp1 = 'foldchanges'
        tmp2 = 'logfoldchanges'
        final_result[tmp1] = 2 ** final_result[tmp2]
        final_result.drop(labels=[tmp2], inplace=True, axis=1)
        final_result.set_index('names', inplace=True)
        final_result['abs_scores'] = np.abs(final_result['scores'])
        final_result.sort_values('abs_scores', ascending=False, inplace=True)
        mydict[perturbation] = final_result
    if fileout is not None:
        import pickle
        fileout.parent.mkdir(parents=True, exist_ok=True)
        with fileout.open('wb') as fout:
            pickle.dump(mydict, fout)
    if return_dict or fileout is None:
        return mydict


SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]


if __name__ == "__main__":
    for dataset in tqdm(chain(SinglePertDataSets, CombPertDataSets)):
        print (dataset)
        calDEG(dataset)
