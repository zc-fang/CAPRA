#coding:utf-8
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

import sys, subprocess, os
from pathlib import Path
from scipy import sparse
from myUtil1 import *
from gears import PertData, GEARS
from gears.inference import evaluate, compute_metrics
import anndata as ad
import torch
import shutil
from itertools import chain

sc.settings.verbosity = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / 'data' / 'datasets'


def get_source_adata_path(dataset_name):
    return DATASETS_ROOT / dataset_name / 'filter_hvg5000_logNor.h5ad'


def get_gears_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / 'hvg5000' / 'GEARS'

def trainModel(adata, issplit = False, seed = 1):
    pert_data = PertData('./data') # specific saved folder   download gene2go_all.pkl
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        pert_data.new_data_process(dataset_name = 'train', adata = adata) # specific dataset name and adata object
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
    if issplit: return
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader

    # set up and train a model
    gears_model = GEARS(pert_data, device = device)
    gears_model.model_initialize(hidden_size = 64)
    gears_model.train(epochs = 15)  ### epochs
    gears_model.save_model('savedModels{}'.format(seed))
    return gears_model

def doGearsFormat(adata):
    def fun1(x):
        if x == 'control': return 'ctrl'
        elif '+' in x:
            genes = x.split('+')
            return genes[0] + '+' + genes[1]
        else: return x + '+' + 'ctrl'
    adata.obs['cell_type'] = 'K562'
    adata.obs['condition'] = adata.obs['perturbation'].apply(lambda x: fun1(x))
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var_names
    if not sparse.issparse(adata.X): adata.X = sparse.csr_matrix(adata.X)
    return adata

def runGears(DataSet, issplit=False, redo=False, seed = 1):
    dirName = get_gears_dir(DataSet)
    if not dirName.is_dir():
        dirName.mkdir(parents=True)
    os.chdir(dirName)
    if redo and os.path.isdir('data/train'):
        shutil.rmtree('data/train')

    if os.path.isfile('savedModels{}/model.pt'.format(seed)): return 
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        adata = sc.read_h5ad(get_source_adata_path(DataSet))
        adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
        adata = doGearsFormat(adata)
        trainModel(adata, issplit = issplit, seed = seed)
    else:
        trainModel(adata='', issplit = issplit, seed = seed)

def loadModel(dirName, seed):
    os.chdir(dirName)
    pert_data = PertData('./data')
    pert_data.load(data_path = './data/train') #
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
    gears_model = GEARS(pert_data, device = device)
    gears_model.load_pretrained('savedModels{}'.format(seed))
    return gears_model

def remove_duplicates_and_preserve_order(input_list):
    deduplicated_dict = OrderedDict.fromkeys(input_list)
    deduplicated_list = list(deduplicated_dict.keys())
    return deduplicated_list

def get_ordered_condition_indices(obs, pert_cats):
    condition_values = obs['condition'].astype(str).to_numpy()
    index_groups = [np.flatnonzero(condition_values == pert_cat) for pert_cat in pert_cats]
    index_groups = [group for group in index_groups if len(group) > 0]
    if len(index_groups) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate(index_groups)

def getPredict(DataSet, seed):
    dirName = get_gears_dir(DataSet)
    os.chdir(dirName)
    result_out = Path(f"savedModels{seed}") / "result.h5ad"
    if result_out.is_file():
        return
    gears_model = loadModel(dirName, seed)
    adata = gears_model.adata
    test_loader = gears_model.dataloader['test_loader']
    test_res = evaluate(test_loader, gears_model.best_model, gears_model.config['uncertainty'], gears_model.device)  
    pert_cats = remove_duplicates_and_preserve_order(test_res['pert_cat'])

    ordered_idx = get_ordered_condition_indices(adata.obs, pert_cats)
    adata2 = adata[ordered_idx].copy()
    adata2.obs['Expcategory'] = 'stimulated'

    pred_matrix = test_res['pred']
    if hasattr(pred_matrix, 'detach'):
        pred_matrix = pred_matrix.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        pred_matrix = np.asarray(pred_matrix, dtype=np.float32)
    adata1 = ad.AnnData(X=pred_matrix, obs=adata2.obs.copy(), var=adata2.var.copy())
    adata1.obs['Expcategory'] = 'imputed'

    adata_ctrl = gears_model.ctrl_adata.copy()
    adata_ctrl.obs['Expcategory'] = 'control'

    del pred_matrix
    del test_res
    del test_loader
    del gears_model

    adata_fi = ad.concat([adata1, adata2, adata_ctrl])
    adata_fi.write(result_out)


seeds = [1, 2, 3, 4, 5]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]

'''
conda activate gears  0.1.0 version
'''


if __name__ == '__main__':
    print ('hello, world')
    # for myDataSet in tqdm(chain(CombPertDataSets, SinglePertDataSets)):
    # for myDataSet in tqdm(['Wessels']):
    #     print (myDataSet)
    #     for seed in seeds:
    #         runGears(myDataSet, issplit=True, redo=False, seed = seed)    ### split data

    for myDataSet in ["Papalexi", "Schmidt"]:
        for seed in tqdm(seeds):
            runGears(myDataSet, issplit=False, redo=False, seed = seed) ### train model
            getPredict(myDataSet, seed)
