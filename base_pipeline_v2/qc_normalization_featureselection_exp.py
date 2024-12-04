import subprocess
from typing import List, Union
import warnings
import scipy.sparse as sp
import pegasus as pg
import pandas as pd
import os
import getpass
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import pegasusio as io
import anndata as ad
import pseudobulk as pb
from pegasusio import UnimodalData, MultimodalData
from typing import Callable, Dict, List
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.stats import f_oneway
from itertools import count
import time
#from adpbulk import ADPBulk
from sklearn.metrics import mean_squared_error
matplotlib.interactive(True)
from matplotlib.backends.backend_pdf import PdfPages
#import phippery as ph
from tqdm import tqdm
import re
#from upsetplot import from_contents
#from upsetplot import UpSet
#from matplotlib.colors import LinearSegmentedColormap
#import scvelo as scv
#import base_pipeline.utils as utils
#import base_pipeline.plotting as plotting
#import sys
#sys.path.append("/projects/home/aantoinette")
from .plotting_exp import create_hexbin_plot
#import .utils_exp as utils
from .utils_exp import *

def check_adata_integrity(adata):
    missing_obs = []
    missing_var = []
    missing_layers = []

    # Check if 'obs' contains required columns
    required_obs = ['sample_id', 'Disease', 'Chemistry', 'Technology', 'Batch']
    for col in required_obs:
        if col not in adata.obs.columns:
            missing_obs.append(col)

    # Check if 'var' contains 'symbol' (gene symbols)
    if 'symbol' not in adata.var.columns:
        missing_var.append('symbol')

    # Check if 'layers' contains 'raw.X'
    if 'raw.X' not in adata.layers:
        missing_layers.append('raw.X')

    # Report missing components to the user
    if missing_obs:
        print(f"Missing columns in 'adata.obs': {', '.join(missing_obs)}. Please add them.")
    else:
        print("All required columns are present in 'adata.obs'.")

    if missing_var:
        print(f"Missing items in 'adata.var': {', '.join(missing_var)}. Please add them.")
    else:
        print("All required items are present in 'adata.var'.")

    if missing_layers:
        print(f"Missing items in 'adata.layers': {', '.join(missing_layers)}. Please add them.")
    else:
        print("All required layers are present in 'adata.layers'.")

    if not missing_obs and not missing_var and not missing_layers:
        print("All required fields are present. No missing components.")




def qc_filtering(data, min_genes, pct_mito,savepath, dataset_name, group):
    """
        This function performs quality control (QC) filtering on a single-cell RNA-seq dataset.
        It calculates basic QC metrics such as the number of genes per cell, the percentage of mitochondrial genes,
        and applies filtering criteria to remove low-quality cells.

        Parameters:
        - data: AnnData object containing the single-cell data.
        - min_genes: Minimum number of genes required for a cell to be retained.
        - pct_mito: Maximum percentage of mitochondrial UMIs allowed for a cell to be retained.
        - savepath: Path where QC plots will be saved.

        Returns:
        - The filtered AnnData object.
        """
    import seaborn as sns

    # Assert that the input data is an AnnData object
    assert isinstance(data, sc.AnnData), "The input data must be an AnnData object."

    #data.obs['Chemistry'] = chemistry
    #data.obs['Condition'] = condition
    os.makedirs(savepath, exist_ok=True)
    data.uns['modality'] = 'rna'
    data.obs['n_genes']=data.X.getnnz(axis=1)
    data.var['mt'] = data.var_names.str.startswith('MT-')
    data.obs['sum_mito_UMIs'] = data.X[:,data.var['mt']].sum(axis=1)
    data.obs['sum_UMIs'] = data.X[:,:].sum(axis=1)
    data.obs['percent_mito'] = data.obs['sum_mito_UMIs']*100/data.obs['sum_UMIs']
    create_hexbin_plot(data, dataset_name=dataset_name, group=group, savepath=savepath, axlines=[min_genes, pct_mito], col_wrap=3)
    sc.pp.filter_cells(data, min_genes=min_genes)
    data = data[data.obs["percent_mito"] <= pct_mito]
    return data



###normalizing
#maybe need an assert statement for raw.X being present in layers?
def normalize(adata):
    # Ensure that 'raw.X' layer exists
    assert 'raw.X' in adata.layers, "'raw.X' layer not found in adata.layers"

    adata.X = adata.layers['raw.X'].copy()
    #print(adata.X[0:10])
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_10e4_counts"] = adata.X.copy()
    return adata



# Feature selection
def feature_selection(adata,group_by,batch_key, bin_path, lineage, save_dir, min_number_cells=5, HVG_min_mean=0.01, HVG_min_disp=0.5):
    print("****** Filtering genes for feature selection...")
    tcr_genes_df = pd.read_csv(os.path.join(bin_path, "TCR_genes.txt"), sep="\t")
    bcr_genes_df = pd.read_csv(os.path.join(bin_path, "BCR_genes.txt"), sep="\t")
    pattern = re.compile("variable|joining|diversity", re.IGNORECASE)
    tcr_genes = tcr_genes_df.loc[tcr_genes_df["Approved name"].str.contains(pattern), "Approved symbol"].tolist()
    bcr_genes = bcr_genes_df.loc[bcr_genes_df["Approved name"].str.contains(pattern), "Approved symbol"].tolist()

    ribo_genes = adata.var["symbol"].loc[adata.var["symbol"].str.startswith(("RPS", "RPL"))].tolist()
    mt_genes = adata.var["symbol"].loc[adata.var["symbol"].str.startswith("MT-")].tolist()

    exclude_genes = set(tcr_genes).union(set(bcr_genes), set(ribo_genes), set(mt_genes))

    adata_forHVG = adata[:, ~adata.var["symbol"].isin(exclude_genes)]

    batch_counts = adata_forHVG.obs[batch_key].value_counts()
    valid_batches = batch_counts[batch_counts >= min_number_cells].index
    adata_forHVG = adata_forHVG[adata_forHVG.obs[batch_key].isin(valid_batches)].copy()

    print("****** Computing highly variable genes...")
    HVGdf = get_HVG(adata_forHVG, groupby=group_by, batch_key=batch_key, min_number_cells=min_number_cells,
                    flavor='seurat', n_top_genes=1500)

    # Save the HVGdf as a CSV file
    hvg_folder = os.path.join(save_dir, "hvg")
    os.makedirs(hvg_folder, exist_ok=True)
    hvg_csv_path = os.path.join(hvg_folder, f"{lineage}_HVG.csv")
    HVGdf.to_csv(hvg_csv_path)

    if group_by is None:
        HVG_selected = HVGdf.index[HVGdf['highly_variable'] == True].tolist()
    else:
        diseases = adata_forHVG.obs.Disease.unique().tolist()
        for disease in diseases:
            hv_col = f'highly_variable_{disease}'
            mean_col = f'means_{disease}'
            disp_col = f'dispersions_{disease}'
            hv_filter_col = f'highly_variable_{disease}_filter'
            HVGdf[hv_filter_col] = HVGdf[hv_col] & (HVGdf[mean_col] >= HVG_min_mean) & (HVGdf[disp_col] >= HVG_min_disp)

        columns_to_keep = [f'highly_variable_{disease}_filter' for disease in adata.obs['Disease'].unique() if
                           f'highly_variable_{disease}_filter' in HVGdf.columns]
        HVG_sel_DF = HVGdf.loc[:, columns_to_keep]
        HVG_sel_DF = HVG_sel_DF[HVG_sel_DF.sum(axis=1) > 0]
        HVG_sel_DF.columns = HVG_sel_DF.columns.str.removeprefix("highly_variable_")
        HVG_selected = HVG_sel_DF.index

    print(f"Total number of genes selected as HV in at least one disease: {len(HVG_selected)}")
    adata.var['highly_variable'] = False
    adata.var.loc[HVG_selected, 'highly_variable'] = True



def remove_genes_for_HVG(adata, TCR=True, BCR=True, ribosomal=True, mitochondrial=True, low_quality=True, sex=True, additional_genes=None, additional_pattern=None):
    """
    Filter genes from the AnnData object to exclude specific gene categories before identifying highly variable genes (HVG).

    Parameters:
    -----------
    adata : AnnData
        An AnnData object containing single-cell gene expression data.
    TCR : bool, optional
        If True, exclude T cell receptor (TCR) genes. Default is True.
    BCR : bool, optional
        If True, exclude B cell receptor (BCR) genes. Default is True.
    ribosomal : bool, optional
        If True, exclude ribosomal genes (RPS, RPL prefixes). Default is True.
    mitochondrial : bool, optional
        If True, exclude mitochondrial genes (MT- prefix). Default is True.
    low_quality : bool, optional
        If True, exclude low_quality-related genes such as MALAT1 and NEAT1. Default is True.
    sex : bool, optional
        If True, exclude sex-related genes such as XIST, RPS4Y1, and DDX3Y. Default is True.
    additional_genes : list, optional
        A list of additional genes to exclude.
    additional_pattern : str, optional
        A pattern to match additional genes for exclusion (e.g., regular expression).

    Returns:
    --------
    adata_forHVG : AnnData
        A filtered AnnData object with specified gene exclusions applied.

    Notes:
    ------
    The function allows flexible gene exclusion based on various categories such as TCR, BCR, ribosomal, mitochondrial,
    and sex-related genes. Users can also specify additional genes or a pattern for further exclusion.
    """

    print("****** Filtering genes for feature selection...")

    exclude_genes = set()

    bin_path = "/projects/blood_cell_atlas/bin"

    # TCR genes
    if TCR:
        tcr_genes_df = pd.read_csv(os.path.join(bin_path, "TCR_genes.txt"), sep="\t")
        pattern = re.compile("variable|joining|diversity", re.IGNORECASE)
        tcr_genes = tcr_genes_df.loc[tcr_genes_df["Approved name"].str.contains(pattern), "Approved symbol"].tolist()
        exclude_genes.update(tcr_genes)

    # BCR genes
    if BCR:
        bcr_genes_df = pd.read_csv(os.path.join(bin_path, "BCR_genes.txt"), sep="\t")
        pattern = re.compile("variable|joining|diversity", re.IGNORECASE)
        bcr_genes = bcr_genes_df.loc[bcr_genes_df["Approved name"].str.contains(pattern), "Approved symbol"].tolist()
        exclude_genes.update(bcr_genes)

    # Ribosomal genes
    if ribosomal:
        ribo_genes = adata.var["symbol"].loc[adata.var["symbol"].str.startswith(("RPS", "RPL"))].tolist()
        exclude_genes.update(ribo_genes)

    # Mitochondrial genes
    if mitochondrial:
        mt_genes = adata.var["symbol"].loc[adata.var["symbol"].str.startswith("MT-")].tolist()
        exclude_genes.update(mt_genes)

    # Low quality genes
    if low_quality:
        lq_genes = ["MALAT1", "NEAT1"]
        exclude_genes.update(lq_genes)

    # Sex genes
    if sex:
        sex_genes = ["XIST", "RPS4Y1", "DDX3Y"]
        exclude_genes.update(sex_genes)

    # Additional genes
    if additional_genes:
        if isinstance(additional_genes, str):
            additional_genes = [additional_genes]
        for add_genes in additional_genes:
            exclude_genes.update(add_genes)

    # Additional pattern-based exclusion
    if additional_pattern:
        if isinstance(additional_pattern, str):
            additional_pattern = [additional_pattern]
        for pattern in additional_pattern:
            pattern_genes = adata.var["symbol"].loc[adata.var["symbol"].str.contains(pattern, na=False)].tolist()
            exclude_genes.update(pattern_genes)

    # Filter the adata object to exclude the specified genes
    adata.var["used_for_HVG"] = True
    adata.var.loc[adata.var["symbol"].isin(exclude_genes), "used_for_HVG"] = False
    adata_forHVG = adata[:, adata.var["used_for_HVG"]]

    return adata_forHVG, exclude_genes