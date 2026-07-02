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
import os, subprocess
import sys
import time
import copy
import shutil
from tqdm import tqdm
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
import anndata as ad
import torch
from scipy import sparse
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab   #type: ignore
from torchtext._torchtext import (   #type: ignore
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
import scanpy as sc
import scgpt as scg  #type: ignore
from scgpt.model import TransformerGenerator   #type: ignore
from scgpt.loss import (   #type: ignore
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch   #type: ignore
from scgpt.tokenizer.gene_tokenizer import GeneVocab   #type: ignore
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics, load_pretrained   #type: ignore

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_SCGPT_MODEL_DIR = REPO_ROOT / "data" / "scgpt"

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
amp = True
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 64
eval_batch_size = 64
schedule_interval = 1
early_stop = 10

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100
load_model = os.environ.get("SCGPT_PRETRAINED_DIR", str(DEFAULT_SCGPT_MODEL_DIR))

'''
https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Perturbation.ipynb
'''

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


def get_source_adata_path(dataset_name):
    return DATASETS_ROOT / dataset_name / "filter_hvg5000_logNor.h5ad"


def get_scgpt_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "scGPT"


def get_gears_splits_dir(dataset_name):
    return DATASETS_ROOT / dataset_name / "hvg5000" / "GEARS" / "data" / "train" / "splits"


def get_seed_output_dir(dataset_name, seed):
    return get_scgpt_dir(dataset_name) / f"savedModels{seed}"


def get_model_output_path(dataset_name, seed):
    return get_seed_output_dir(dataset_name, seed) / "best_model.pt"


def get_result_output_path(dataset_name, seed):
    return get_seed_output_dir(dataset_name, seed) / "result.h5ad"


def is_valid_h5ad(path):
    if not path.is_file():
        return False
    try:
        backed = ad.read_h5ad(path, backed="r")
        backed.file.close()
    except Exception as exc:
        print(f"[redo predict] existing h5ad is not readable: {path} ({exc})")
        return False
    return True


def make_h5ad_dataframe_index_safe(adata):
    for attr in ("obs", "var"):
        df = getattr(adata, attr)
        if df.index.name is not None and df.index.name in df.columns:
            df = df.copy()
            if attr == "var" and df.index.name == "gene_name":
                df.drop(columns=[df.index.name], inplace=True)
            else:
                df.index.name = None
            setattr(adata, attr, df)
    return adata


def remove_duplicates_and_preserve_order(values):
    return list(dict.fromkeys(str(value) for value in values))


def get_ordered_condition_indices(obs, pert_cats):
    condition_values = obs["condition"].astype(str).to_numpy()
    index_groups = []
    missing = []
    for pert_cat in remove_duplicates_and_preserve_order(pert_cats):
        group = np.flatnonzero(condition_values == pert_cat)
        if len(group) == 0:
            missing.append(pert_cat)
        else:
            index_groups.append(group)
    if missing:
        raise ValueError(f"Missing predicted conditions in adata.obs['condition']: {missing[:10]}")
    if len(index_groups) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate(index_groups)


def write_h5ad_atomic(adata, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        adata.write(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise



def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device, gene_ids
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred_batches = []
    truth_batches = []
    pred_de = []
    truth_de = []
    results = {}

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.inference_mode():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            p_cpu = p.detach().cpu()
            t_cpu = t.detach().cpu()
            pred_batches.append(p_cpu.numpy().astype(np.float32, copy=False))
            truth_batches.append(t_cpu.numpy().astype(np.float32, copy=False))

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p_cpu[itr, de_idx].numpy().astype(np.float32, copy=False))
                truth_de.append(t_cpu[itr, de_idx].numpy().astype(np.float32, copy=False))

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    results["pred"] = np.concatenate(pred_batches, axis=0).astype(float, copy=False)
    results["truth"] = np.concatenate(truth_batches, axis=0).astype(float, copy=False)

    results["pred_de"] = np.stack(pred_de, axis=0).astype(float, copy=False)
    results["truth_de"] = np.stack(truth_de, axis=0).astype(float, copy=False)

    return results

def train_scGPT(DataSet, istrain=True, seed = 1):
    def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        total_loss, total_mse = 0.0, 0.0
        start_time = time.time()

        num_batches = len(train_loader)
        for batch, batch_data in enumerate(train_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                else:
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )
                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=device
                )

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool
                )  # Use all
                loss = loss_mse = criterion(output_values, target_values, masked_positions)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            # torch.cuda.empty_cache()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                )
                total_loss = 0
                total_mse = 0
                start_time = time.time()

    dirName = get_scgpt_dir(DataSet)
    if not dirName.is_dir():
        dirName.mkdir(parents=True)
    os.chdir(dirName)

    modeloutPT = get_model_output_path(DataSet, seed)
    if modeloutPT.is_file() and istrain:
        print(f"[skip train] {DataSet} seed {seed}: found {modeloutPT}")
        return   ### 已经跑过模型就不需要重新跑了
    print (DataSet)
    logger = scg.logger
    scg.utils.add_file_handler(logger,  "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

    adata = sc.read_h5ad(get_source_adata_path(DataSet))
    adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
    adata = doGearsFormat(adata)
    pert_data = PertData('./data') # specific saved folder   download gene2go_all.pkl
    pert_data.new_data_process(dataset_name = 'train', adata = adata) # specific dat
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
    
    tmp_dir1 = dirName / 'data' / 'train' / 'splits'
    tmp_dir2 = get_gears_splits_dir(DataSet)
    if not tmp_dir1.is_dir():
        tmp_dir1.mkdir(parents=True)
    if tmp_dir2.is_dir():
        for split_file in tmp_dir2.iterdir():
            if split_file.is_file():
                shutil.copy2(split_file, tmp_dir1 / split_file.name)
    
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    model_dir = Path(load_model)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"scGPT pretrained model directory not found: {model_dir}. "
            "Set SCGPT_PRETRAINED_DIR to the directory containing args.json, vocab.json, and best_model.pt."
        )
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
    model_dropout = model_configs.get("dropout", dropout)
    model_use_fast_transformer = model_configs.get("fast_transformer", use_fast_transformer)
    model_pre_norm = model_configs.get("pre_norm", False)


    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=model_dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=model_use_fast_transformer,
        pre_norm=model_pre_norm,
    )

    pretrained_dict = torch.load(model_file)
    model = load_pretrained(
        model,
        pretrained_dict,
        strict=False,
        prefix=load_param_prefixs,
        verbose=True,
    )

    model.to(device)

    if not istrain:  return pert_data, model, gene_ids  #### 进行预测


    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val_loss = float("inf")
    best_val_corr = 0
    best_model = None
    patience = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train(
            model,
            train_loader,
        )

        val_res = eval_perturb(valid_loader, model, device, gene_ids)
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break
        scheduler.step()
    modeloutPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), modeloutPT)


def doPredict(DataSet, seed):
    resultOut = get_result_output_path(DataSet, seed)
    if is_valid_h5ad(resultOut):
        print(f"[skip predict] {DataSet} seed {seed}: found {resultOut}")
        return

    os.chdir(get_scgpt_dir(DataSet))
    filein = get_model_output_path(DataSet, seed)
    if not filein.is_file():
        raise FileNotFoundError(f"scGPT trained model not found for prediction: {filein}")
    pert_data, model, gene_ids = train_scGPT(DataSet, istrain=False, seed = seed)
    model.load_state_dict(torch.load(filein, map_location=device))
    test_loader = pert_data.dataloader["test_loader"]
    test_res = eval_perturb(test_loader, model, device, gene_ids)
    adata = pert_data.adata
    
    ordered_idx = get_ordered_condition_indices(adata.obs, test_res['pert_cat'])
    adata_truth = adata[ordered_idx].copy()
    adata_truth.obs['Expcategory'] = 'stimulated'

    pred_matrix = np.asarray(test_res['pred'], dtype=np.float32)
    if pred_matrix.shape[0] != adata_truth.n_obs:
        raise ValueError(
            f"Prediction/truth row mismatch: pred={pred_matrix.shape[0]}, "
            f"truth={adata_truth.n_obs}"
        )
    adata_pred = ad.AnnData(X=pred_matrix, obs=adata_truth.obs.copy(), var=adata_truth.var.copy())
    adata_pred.obs['Expcategory'] = 'imputed'

    adata_ctrl = adata[adata.obs['perturbation'].isin(['control'])].copy()
    adata_ctrl.obs['Expcategory'] = 'control'

    if sparse.issparse(adata_truth.X):
        adata_truth.X = adata_truth.X.toarray().astype(np.float32, copy=False)
    else:
        adata_truth.X = np.asarray(adata_truth.X, dtype=np.float32)

    if sparse.issparse(adata_ctrl.X):
        adata_ctrl.X = adata_ctrl.X.toarray().astype(np.float32, copy=False)
    else:
        adata_ctrl.X = np.asarray(adata_ctrl.X, dtype=np.float32)

    del pred_matrix
    del test_loader
    del test_res
    del model
    del pert_data

    result = ad.concat([adata_pred, adata_truth, adata_ctrl], merge="same")
    result = make_h5ad_dataframe_index_safe(result)
    write_h5ad_atomic(result, resultOut)

'''
conda activate scGPT
'''

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]
epochs = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(42)
seeds = [1, 2, 3, 4, 5]

if __name__ == '__main__':
    print ('hello, world')
    for myDataSet in chain(["Replogle_K562essential"]):
        for seed in seeds:
            modeloutPT = get_model_output_path(myDataSet, seed)
            resultOut = get_result_output_path(myDataSet, seed)

            if is_valid_h5ad(resultOut):
                print(f"[skip complete] {myDataSet} seed {seed}: found {resultOut}")
                continue

            if modeloutPT.is_file():
                print(f"[skip train] {myDataSet} seed {seed}: found {modeloutPT}")
            else:
                train_scGPT(myDataSet, istrain=True, seed = seed)

            doPredict(myDataSet, seed = seed)
