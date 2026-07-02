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

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
R_SCRIPT_PATH = SCRIPT_DIR / "linearModel.r"
CONDA_EXE = os.environ.get("CONDA_EXE", "conda")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from myUtil1 import *
import scanpy as sc
from tqdm import tqdm
from gears import PertData

warnings.filterwarnings('ignore')


def get_linear_model_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "linearModel"


def get_gears_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "GEARS"


def get_train_mean_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "trainMean"

def normalize_condition_names(obs):
  import pandas as pd
  ser = pd.Series(["+".join(sorted(x.split("+"))) for x in obs['condition']], dtype = "category", index = obs.index)
  obs['condition'] = ser
  return obs


def preData1(DataSet, seed, isComb=False):
    dirName = get_linear_model_dir(DataSet)
    dirName.mkdir(parents=True, exist_ok=True)
    os.chdir(dirName)

    target_data_dir = dirName / "data"
    if target_data_dir.is_dir():
        shutil.rmtree(target_data_dir)
    shutil.copytree(get_gears_dir(DataSet) / "data", target_data_dir)

    if isComb:
        pert_data = PertData('./data') # specific saved folder   download gene2go_all.pkl
        pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
        pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
        norman_adata = pert_data.adata
        new_obs = normalize_condition_names(norman_adata.obs.copy())
        if not norman_adata.obs.equals(new_obs):
            norman_adata.obs = new_obs
            # Override the perturb_processed.h5ad
            norman_adata.write_h5ad("data/train/perturb_processed.h5ad")
            # Delete the data_pyg folder because it has the problematic references to the 
            data_pyg_folder = Path("data/train/data_pyg")
            if data_pyg_folder.exists():
                (data_pyg_folder / "cell_graphs.pkl").unlink(missing_ok = True)
                data_pyg_folder.rmdir()
                pert_data = PertData('./data') # specific saved folder
                pert_data.load(data_path = './data/train') # load the processed data,
        conds = norman_adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()
        single_pert = [x for x in conds if 'ctrl' in x]
        double_pert = np.setdiff1d(conds, single_pert).tolist()
        double_training = np.random.choice(double_pert, size=len(double_pert) // 2, replace=False).tolist()
        double_test = np.setdiff1d(double_pert, double_training).tolist()
        double_test = double_test[0:(len(double_test)//2)]
        double_holdout = np.setdiff1d(double_pert, double_training + double_test).tolist()
        set2conditions = {   #### 组合扰动50% is  double_training, 25% is  double_test, 25% is double_holdout
            "train": single_pert + double_training,
            "test": double_test,
            "val": double_holdout
        }
    else:
        pert_data = PertData('./data')
        pert_data.load(data_path = './data/train') # load the
        pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8)
        set2conditions = pert_data.set2conditions

    outfile = dirName / "data" / "train" / "splits" / f"set2conditions_{seed}.tsv"
    with open(outfile, "w") as fout:
        json.dump(set2conditions, fout)


def runLinearModel(DataSet, seed):
    dirName = get_linear_model_dir(DataSet)
    dirName.mkdir(parents=True, exist_ok=True)
    os.chdir(dirName)
    dirOut = dirName / f"savedModels{seed}"
    dirOut.mkdir(parents=True, exist_ok=True)
    cmd = [CONDA_EXE, "run", "-n", "linear_perturbation_prediction", "Rscript", str(R_SCRIPT_PATH), str(seed), str(dirName)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


### 根据预测的生成表达量
def generateH5ad(DataSet, seed = 1):
    dirName = get_linear_model_dir(DataSet)
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


def predComb(DataSet, seed):
    dirName = get_linear_model_dir(DataSet)
    os.chdir(dirName)
    filein1 = dirName / f"savedModels{seed}" / "pred.tsv"
    filein2 = get_train_mean_dir(DataSet) / f"savedModels{seed}" / "pred.tsv"
    dat1 = pd.read_csv(filein1, sep='\t', index_col=0).T
    dat2 = pd.read_csv(filein2, sep='\t', index_col=0)
    expGene = list(np.intersect1d(dat1.columns, dat2.columns))
    single_perts = [i for i in dat1.index if '+' not in i and i != 'ctrl']
    comb_perts = [i for i in dat2.index if '+' in i]
    dat1 = dat1.loc[single_perts, expGene]; dat2 = dat2.loc[comb_perts, expGene]
    pred = pd.concat([dat1, dat2])
    dirOut = dirName / f"savedModels{seed}"
    pred.to_csv(dirOut / "pred.tsv", sep='\t')

'''
conda activate linear_perturbation_prediction
'''

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]
seeds = [1, 2, 3, 4, 5]

if __name__ == '__main__':
    print ('hello, world')
    for DataSet in tqdm(["Replogle_K562essential", "Papalexi", "Schmidt"]):
        print (DataSet)
        for seed in tqdm(seeds):
            preData1(DataSet, seed, isComb=False)
            runLinearModel(DataSet, seed)
            generateH5ad(DataSet, seed)

"""    for DataSet in tqdm(CombPertDataSets):
        print (DataSet)
        for seed in tqdm(seeds):
            preData1(DataSet, seed, isComb=True)
            runLinearModel(DataSet, seed)
            predComb(DataSet, seed)
            generateH5ad(DataSet, seed)"""
