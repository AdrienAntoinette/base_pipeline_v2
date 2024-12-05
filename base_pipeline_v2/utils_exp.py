import pandas as pd
from pegasusio import UnimodalData

import pseudobulk as pb
from matplotlib.backends.backend_pdf import PdfPages
#import scib
#import anndata2ri
import os
from scanpy.external.pp import harmony_integrate
import scib_metrics
import scanpy as sc
import pegasus as pg
from harmony import harmonize
from scanpy.external.pp import scanorama_integrate
from scanpy.external.pp import bbknn
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import re
import seaborn as sns
from typing import Dict
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
from anndata import AnnData

def compute_metrics(X, Labels, Batches, adata, ksim_attr, ksim_rep):
    """
    Compute a series of clustering and batch correction metrics.

    Parameters:
    - X: np.ndarray, embedding or feature space data matrix.
    - Labels: list or np.ndarray, true labels for each point in X.
    - Batches: list or np.ndarray, batch information for each point in X.
    - adata: AnnData, annotated data object.
    - ksim_attr: attribute for kSIM calculation.
    - ksim_rep: representation for kSIM calculation.

    Returns:
    - metrics: dict, containing ASW_l, ASW_b, ARI, NMI, ilisi, clisi, kSIM, and kBet.
    """

    # Assertions to check input types
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(Labels, (list, np.ndarray)), "Labels should be a list or numpy array."
    assert isinstance(Batches, (list, np.ndarray)), "Batches should be a list or numpy array."
    assert isinstance(adata, AnnData), "adata should be an AnnData object."

    # Calculate ASW_l (Silhouette Score for Labels)
    print("Calculating ASW_l (Silhouette Score for Labels)...")
    ASW_l = scib_metrics.silhouette_label(X, labels=Labels)
    print(f"ASW_l: {ASW_l}")

    # Calculate ASW_b (Silhouette Score for Batches)
    print("Calculating ASW_b (Silhouette Score for Batches)...")
    ASW_b = scib_metrics.silhouette_batch(X, labels=Labels, batch=Batches)
    print(f"ASW_b: {ASW_b}")

    # Calculate ARI and NMI using k-means clustering
    print("Calculating ARI and NMI using k-means clustering...")
    ARI_NMI = scib_metrics.nmi_ari_cluster_labels_kmeans(X, Labels)
    ARI = ARI_NMI['ari']
    NMI = ARI_NMI['nmi']
    print(f"ARI: {ARI}, NMI: {NMI}")

    # Perform nearest neighbor search for ilisi and clisi metrics
    print("Performing nearest neighbor search for ilisi and clisi metrics...")
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Nearest neighbor results for ilisi and clisi
    neigh = scib_metrics.nearest_neighbors.NeighborsResults(indices, distances)

    # Calculate ilisi
    print("Calculating ilisi...")
    ilisi = scib_metrics.ilisi_knn(neigh, batches=Batches)
    print(f"ilisi: {ilisi}")

    # Calculate clisi
    print("Calculating clisi...")
    clisi = scib_metrics.clisi_knn(neigh, labels=Labels, scale=True)
    print(f"clisi: {clisi}")

    # Calculate kBET metric
    print("Calculating kBET metric...")
    kbet, _, _ = scib_metrics.kbet(neigh, batches=Batches)
    print(f"kBET: {kbet}")

    # Calculate kSIM metric
    print("Calculating kSIM metric...")
    adata_mul = io.MultimodalData(adata)
    _, kSIM = pg.calc_kSIM(adata_mul, attr=ksim_attr, rep=ksim_rep)
    print(f"kSIM: {kSIM}")

    # Return metrics dictionary
    metrics = {
        'ASW_l': ASW_l,
        'ASW_b': ASW_b,
        'ARI': ARI,
        'NMI': NMI,
        'ilisi': ilisi,
        'clisi': clisi,
        'kSIM': kSIM,
        'kBet': kbet
    }

    return metrics


import os
import pandas as pd
import scanpy as sc
from anndata import AnnData


def run_marker_genes_analysis(adata, de_key, save_dir, leiden_cluster, logFC_thr=0.25, pval_thr=0.05, pct_nz_thr=0.25,
                              save_excel=True, visualize_markers=False):
    """
    Perform marker gene analysis, save results in CSV/Excel, and visualize top marker genes.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - de_key: str, key to store DE results in adata.
    - save_dir: str, directory path to save results.
    - leiden_cluster: str, column in adata.obs used for clustering/grouping cells.
    - logFC_thr: float, log-fold change threshold for filtering genes.
    - pval_thr: float, adjusted p-value threshold for filtering genes.
    - pct_nz_thr: float, minimum percentage of non-zero expression in group.
    - save_excel: bool, save results in Excel format.
    - visualize_markers: bool, generate plots for top marker genes.

    Returns:
    - Updated adata object with marker gene information in varm attribute.
    """

    # Assert statements to validate inputs
    assert isinstance(adata, AnnData), "adata must be an AnnData object."
    assert isinstance(de_key, str) and de_key, "de_key must be a non-empty string."
    assert os.path.isdir(save_dir), f"save_dir '{save_dir}' does not exist."
    assert 0 <= logFC_thr <= 10, "logFC_thr should be between 0 and 10."
    assert 0 <= pval_thr <= 1, "pval_thr should be between 0 and 1."
    assert 0 <= pct_nz_thr <= 1, "pct_nz_thr should be between 0 and 1."

    print(f'Running marker genes analysis for {de_key}...')

    # Run differential expression analysis with the rank_genes_groups method
    print("Performing rank genes groups analysis...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=leiden_cluster,
        method='wilcoxon',
        pts=True,
        key_added=de_key,
        use_raw=False,
    )

    # Convert differential expression results into a DataFrame and save as CSV
    rank_genesDF = sc.get.rank_genes_groups_df(adata, group=None, key=de_key, gene_symbols="symbol")
    csv_path = os.path.join(save_dir, f'{de_key}_markers.csv')
    rank_genesDF.to_csv(csv_path, index=False)
    print(f"DE results saved as CSV at: {csv_path}")

    # Create a pivoted DataFrame for storage in adata.varm
    pivoted_df = rank_genesDF.pivot(index='names', columns='group')
    pivoted_df.columns = [f'{group}::{col}' for col, group in pivoted_df.columns]
    pivoted_df = pivoted_df.reindex(adata.var.index)
    adata.varm[de_key] = pivoted_df.to_records(index=False)
    print("Pivoted DataFrame created and stored in adata.varm.")

    # Save filtered results in an Excel file if requested
    if save_excel:
        print("Saving filtered results to Excel...")
        rank_genes_excel_path = os.path.join(save_dir, f'{de_key}_markers_filtered.xlsx')
        unique_groups = rank_genesDF["group"].unique()
        with pd.ExcelWriter(rank_genes_excel_path, engine="openpyxl") as writer:
            for group in unique_groups:
                # Filter DataFrame based on thresholds for logFC, adjusted p-value, and non-zero percentage
                group_rank_genesDF = rank_genesDF[rank_genesDF["group"] == group]
                group_rank_genesDF = group_rank_genesDF[group_rank_genesDF["logfoldchanges"] > logFC_thr]
                group_rank_genesDF = group_rank_genesDF[group_rank_genesDF["pvals_adj"] < pval_thr]
                group_rank_genesDF = group_rank_genesDF[group_rank_genesDF["pct_nz_group"] > pct_nz_thr]
                group_rank_genesDF = group_rank_genesDF.groupby("group", group_keys=False).apply(
                    lambda x: x.sort_values(by="logfoldchanges", ascending=False)
                )
                group_rank_genesDF.to_excel(writer, sheet_name=group, index=False)
        print(f"Filtered DE results saved to Excel at: {rank_genes_excel_path}")

    # Optionally visualize the markers
    if visualize_markers:
        print("Visualizing top marker genes...")
        sc._settings.settings._vector_friendly = True  # Set vector-friendly option for plotting
        visualize_top_DEmarkers(
            adata, rank_genesDF, leiden_cluster=leiden_cluster,
            order_by="logfoldchanges", save_dir=save_dir, prefix="DE_",
            top_markers_per_group=20, logFC_thr=0.25, pval_thr=0.05, pct_nz_thr=0.25
        )
        print("Marker visualization complete.")

    return adata  # Return the modified adata object with updated varm attribute




def visualize_top_DEmarkers(adata, markers_df, leiden_cluster, save_dir, prefix, order_by, top_markers_per_group=5,
                            logFC_thr=0.25, pval_thr=0.05, pct_nz_thr=0.25, max_cells_plot=100000):
    # Iterate through each group in markers_df
    for group in markers_df['group'].unique():
        group_rank_genesDF = markers_df[markers_df["group"] == group]
        group_rank_genesDF = group_rank_genesDF[group_rank_genesDF["logfoldchanges"] > logFC_thr]
        group_rank_genesDF = group_rank_genesDF[group_rank_genesDF["pvals_adj"] < pval_thr]
        group_rank_genesDF = group_rank_genesDF[group_rank_genesDF["pct_nz_group"] > pct_nz_thr]
        group_rank_genesDF = group_rank_genesDF.groupby("group", group_keys=False).apply(
            lambda x: x.sort_values(by=order_by, ascending=False), include_groups=False)

        top_markers = group_rank_genesDF.head(top_markers_per_group)['names'].tolist()

        if not top_markers:
            print(f"No markers passed the filters for group {group}. Skipping plots for this group.")
            continue

        # Plot UMAP for top markers
        sc._settings.settings._vector_friendly = True
        fraction = min(1, max_cells_plot / adata.shape[0])
        random_indices = balanced_sample(adata.obs, cols=leiden_cluster, frac=fraction, shuffle=True,
                                         random_state=42).CellID
        fig_umap = sc.pl.embedding(adata[random_indices], basis='X_umap', vmin='.5', vmax='p99',
                                   color=top_markers, wspace=0.4, ncols=5, return_fig=True)
        plt.savefig(os.path.join(save_dir, f"{prefix}_{group}_top{top_markers_per_group}_markers_UMAP.pdf"),
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig_umap)

        # Plot DotPlot for top markers
        sc.pl.dotplot(adata[random_indices, :], var_names=top_markers, groupby=leiden_cluster,
                      standard_scale='var',
                      use_raw=False, dendrogram=False, show=False, return_fig=True).add_totals().style(
            dot_edge_color='black', dot_edge_lw=0.25, cmap="Reds").savefig(
            os.path.join(save_dir, f"{prefix}_{group}_top{top_markers_per_group}_markers_DotPlot.pdf"),
            bbox_inches='tight', pad_inches=0, dpi=300)



def get_HVG(adata, groupby=None, batch_key=None, flavor='seurat', min_number_cells=5, n_top_genes=3000, n_bins=20):
    import anndata as ad
    import pandas as pd
    import numpy as np
    import scanpy as sc

    HVGdf = pd.DataFrame()

    if groupby is not None:
        listGroups = adata.obs[groupby].unique()

        for g in listGroups:
            print(g)
            # Explicitly create a copy to avoid ImplicitModificationWarning
            adata_g = adata[adata.obs[groupby] == g].copy()

            # Check if all {batch_key} categories in each {g} have at least min_number_cells
            if batch_key is not None:
                batch_counts = adata_g.obs[batch_key].value_counts()
                if all(batch_counts >= min_number_cells):
                    # All categories have enough cells
                    HVGdf_i = sc.pp.highly_variable_genes(adata=adata_g,
                                                          batch_key=batch_key,
                                                          flavor=flavor,
                                                          n_top_genes=n_top_genes,
                                                          n_bins=n_bins,
                                                          inplace=False)
                else:
                    # Filter out categories with less than min_number_cells
                    valid_batches = batch_counts[batch_counts >= min_number_cells].index
                    adata_g_filtered = adata_g[adata_g.obs[batch_key].isin(valid_batches)].copy()

                    # Calculate the percentage of categories removed
                    num_removed = len(batch_counts) - len(valid_batches)
                    total_batches = len(batch_counts)
                    percent_removed = (num_removed / total_batches) * 100

                    if len(valid_batches) >= 0.1 * total_batches:
                        # If more than 10% of the categories are valid, proceed
                        print(f"WARNING: Removed {percent_removed:.2f}% ({num_removed}/{total_batches}) "
                              f"of {batch_key} categories for group {g}.")
                        HVGdf_i = sc.pp.highly_variable_genes(adata=adata_g_filtered,
                                                              batch_key=batch_key,
                                                              flavor=flavor,
                                                              n_top_genes=n_top_genes,
                                                              n_bins=n_bins,
                                                              inplace=False)
                    else:
                        # If less than 10% of categories are valid, run without batch_key
                        print(f"WARNING: Removed {percent_removed:.2f}% ({num_removed}/{total_batches}) "
                              f"of {batch_key} categories for group {g}. "
                              f"Removing batch_key due to insufficient valid categories.")
                        HVGdf_i = sc.pp.highly_variable_genes(adata=adata_g,
                                                              batch_key=None,
                                                              flavor=flavor,
                                                              n_top_genes=n_top_genes,
                                                              n_bins=n_bins,
                                                              inplace=False)
            else:
                HVGdf_i = sc.pp.highly_variable_genes(adata=adata_g,
                                                      batch_key=batch_key,
                                                      flavor=flavor,
                                                      n_top_genes=n_top_genes,
                                                      n_bins=n_bins,
                                                      inplace=False)

            HVGdf_i = HVGdf_i.add_suffix('_{}'.format(g))
            if batch_key is None:
                HVGdf_i['gene_name'] = adata_g.var_names
                HVGdf_i.set_index('gene_name', inplace=True, drop=True)
                HVGdf_i.index.name = None

            # Merge the current group's HVG dataframe with the main one
            HVGdf = HVGdf.merge(HVGdf_i, how='right', left_index=True, right_index=True)

    else:
        # When groupby is None, check if the entire dataset meets the min_number_cells threshold
        if batch_key is not None:
            batch_counts = adata.obs[batch_key].value_counts()
            if all(batch_counts >= min_number_cells):
                # All categories have enough cells
                HVGdf = sc.pp.highly_variable_genes(adata=adata,
                                                    batch_key=batch_key,
                                                    flavor=flavor,
                                                    n_top_genes=n_top_genes,
                                                    n_bins=n_bins,
                                                    inplace=False)
            else:
                # Filter out categories with less than min_number_cells
                valid_batches = batch_counts[batch_counts >= min_number_cells].index
                adata_filtered = adata[adata.obs[batch_key].isin(valid_batches)].copy()

                # Calculate the percentage of categories removed
                num_removed = len(batch_counts) - len(valid_batches)
                total_batches = len(batch_counts)
                percent_removed = (num_removed / total_batches) * 100

                if len(valid_batches) >= 0.1 * total_batches:
                    # If more than 10% of the categories are valid, proceed
                    print(f"WARNING: Removed {percent_removed:.2f}% ({num_removed}/{total_batches}) "
                          f"of {batch_key} categories.")
                    HVGdf = sc.pp.highly_variable_genes(adata=adata_filtered,
                                                        batch_key=batch_key,
                                                        flavor=flavor,
                                                        n_top_genes=n_top_genes,
                                                        n_bins=n_bins,
                                                        inplace=False)
                else:
                    # If less than 10% of categories are valid, run without batch_key
                    print(f"WARNING: Removed {percent_removed:.2f}% ({num_removed}/{total_batches}) "
                          f"of {batch_key} categories. Removing batch_key due to insufficient valid categories.")
                    HVGdf = sc.pp.highly_variable_genes(adata=adata,
                                                        batch_key=None,
                                                        flavor=flavor,
                                                        n_top_genes=n_top_genes,
                                                        n_bins=n_bins,
                                                        inplace=False)
        else:
            HVGdf = sc.pp.highly_variable_genes(adata=adata,
                                                batch_key=batch_key,
                                                flavor=flavor,
                                                n_top_genes=n_top_genes,
                                                n_bins=n_bins,
                                                inplace=False)

        HVGdf['gene_name'] = adata.var_names
        HVGdf.set_index('gene_name', inplace=True, drop=True)
        HVGdf.index.name = None

    return HVGdf


def balanced_sample(df, cols=None, n=None, frac=None, shuffle=False, random_state=42):
    import pandas as pd
    if not ((n is None) != (frac is None)):
        print("Error: please specify n or frac, not both")
        return None

    # Group by the columns and apply the sample function
    df["CellID"] = df.index
    if cols is None:
        df_sampled = df.apply(lambda x: x.sample(n=n, frac=frac, replace=False, random_state=random_state))
    else:
        df_sampled = df.groupby(cols, observed=True).apply(
            lambda x: x.sample(n=n, frac=frac, replace=False, random_state=random_state))
        df_sampled = df_sampled.drop(cols, axis=1, errors='ignore').reset_index(drop=False)

    if shuffle:
        return df_sampled.sample(frac=1, random_state=random_state)
    else:
        return df_sampled

#
# def run_mcv_functions_sc(adata, savepath, iteration='mcv', figsize=(10, 3), raw_name='X', hvf_name='highly_variable'):
#     def recipe_scanpy(adata, batch_key=None, pcs=50):
#         # Normalize and log-transform the data
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#         sc.pp.highly_variable_genes(adata)
#
#         # Run PCA and batch correction if specified
#         if pcs:
#             sc.tl.pca(adata, n_comps=pcs)
#         if batch_key:
#             sc.external.pp.harmony_integrate(adata, batch_key=batch_key)
#
#     def mcv_calibrate_pca(adata, selected, max_pcs=100):
#         X = adata.layers[raw_name] if raw_name in adata.layers else adata.X
#
#         # Split into two parts for MCV calculation
#         data1 = np.random.binomial(X.data, 0.5)
#         data2 = X.data - data1
#         assert np.all(X.data == data1 + data2)
#
#         X1 = csr_matrix((data1, X.indices, X.indptr), shape=X.shape)
#         X2 = csr_matrix((data2, X.indices, X.indptr), shape=X.shape)
#
#         adata1 = adata[:, selected].copy()
#         adata2 = adata[:, selected].copy()
#         adata1.X = X1
#         adata2.X = X2
#
#         recipe_scanpy(adata1, batch_key='batch' if 'batch' in adata.obs.columns else None)
#         recipe_scanpy(adata2, batch_key='batch' if 'batch' in adata.obs.columns else None)
#
#         X1_dense = adata1.X.todense()
#         X2_dense = adata2.X.todense()
#
#         sc.tl.pca(adata1, n_comps=max_pcs)
#         k_range = np.concatenate([np.arange(2, 10, 1), np.arange(10, 30, 2), np.arange(30, max_pcs, 5)])
#
#         mcv_loss = np.zeros(len(k_range))
#         rec_loss = np.zeros(len(k_range))
#
#         for i, k in enumerate(tqdm(k_range)):
#             reconstruction = adata1.obsm['X_pca'][:, :k] @ adata1.uns['pca']['components_'][:k, :]
#             mcv_loss[i] = mean_squared_error(reconstruction, X2_dense)
#             rec_loss[i] = mean_squared_error(reconstruction, X1_dense)
#
#         optimal_k = k_range[np.argmin(mcv_loss)]
#
#         mcv_summary = {
#             'k_range': k_range,
#             'mcv_loss': mcv_loss,
#             'rec_loss': rec_loss
#         }
#         return optimal_k, mcv_summary
#
#     def plot_mcv_pca(mcv_summary, figdir, save_name):
#         plt.figure(figsize=figsize)
#         plt.plot(mcv_summary['k_range'], mcv_summary['mcv_loss'])
#         optimal_k = mcv_summary['k_range'][np.argmin(mcv_summary['mcv_loss'])]
#         plt.scatter([optimal_k], [mcv_summary['mcv_loss'][np.argmin(mcv_summary['mcv_loss'])]], c='k')
#         plt.xlabel('Number of PCs')
#         plt.ylabel('MCV Loss')
#         plt.title(f'Optimal PCs = {optimal_k}')
#         plt.tight_layout()
#         plt.savefig(os.path.join(figdir, f"{save_name}_pca_mcv.png"))
#         plt.close()
#
#     # Run calibration and plot
#     selected_genes = adata.var_names[adata.var[hvf_name]]
#     optimal_k, mcv_summary = mcv_calibrate_pca(adata, selected=selected_genes, max_pcs=100)
#     plot_mcv_pca(mcv_summary, figdir=savepath, save_name=iteration)
#
#     return optimal_k

def run_mcv_functions_sc(mcv_data, savepath, iteration= 'mcv', figsize=[10,3], raw_name= 'raw.X',  hvf_name= 'highly_variable_features'):
    def recipe_pegasus(unidata: UnimodalData, channel_name: str = None, pcs: int = None):
        # Courtesy of Tos Chan
        pg.identify_robust_genes(unidata)
        pg.log_norm(unidata)
        pg.highly_variable_features(unidata)
        if pcs is not None and channel_name is not None:
            pg.pca(unidata, n_components=pcs, random_state=1)
            pg.run_harmony(unidata, batch= channel_name)
            pg.neighbors(unidata)
            print('PCA, Harmony, and KNN finished. rep_key = pca_harmony')

    def mcv_calibrate_pca(multidata, selected: List, max_pcs: int = 100, raw_name= raw_name, hvf_name= 'highly_variable_features'):
        # Courtesy of Tos Chan
        # assert type(multidata) == io.UnimodalData or type(multidata) == io.multimodal_data.MultimodalData
        multidata = io.MultimodalData(multidata)

        X = multidata.get_matrix(raw_name)

        data = X.data.astype(np.int64)
        indptr = X.indptr
        indices = X.indices

        data1 = np.random.binomial(data, 0.5)
        data2 = data - data1
        assert np.all(data == data1 + data2)

        X1 = csr_matrix((data1.astype(np.float32), indices, indptr), shape=X.shape)
        X2 = csr_matrix((data2.astype(np.float32), indices, indptr), shape=X.shape)

        del X, data1, data2, indptr, indices

        adata1 = io.UnimodalData(barcode_metadata=multidata.obs, feature_metadata=multidata.var, matrices={'X': X1}, modality='rna')
        adata2 = io.UnimodalData(barcode_metadata=multidata.obs, feature_metadata=multidata.var, matrices={'X': X2}, modality='rna')

        recipe_pegasus(adata1, channel_name='Channel')
        recipe_pegasus(adata2, channel_name='Channel')

        adata1 = adata1[:, selected].copy()
        adata2 = adata2[:, selected].copy()

        X1 = adata1.X.todense()
        X2 = adata2.X.todense()

        del adata2

        max_pcs = np.maximum(max_pcs, 100)

        adata1.var[hvf_name] = True
        pg.pca(adata1, n_components=max_pcs, random_state=1)

        k_range = np.concatenate([np.arange(2, 10, 1), np.arange(10, 30, 2), np.arange(30, max_pcs, 5)])

        mcv_loss = np.zeros(len(k_range))
        rec_loss = np.zeros(len(k_range))

        for i, k in enumerate(tqdm(k_range)):
            reconstruction = adata1.obsm['X_pca'][:, :k].dot(adata1.uns['PCs'].T[:k])
            mcv_loss[i] = mean_squared_error(np.asarray(reconstruction), np.asarray(X2))
            rec_loss[i] = mean_squared_error(np.asarray(reconstruction), np.asarray(X1))

        optimal_k = k_range[np.argmin(mcv_loss)]

        mcv_summary = {'k_range': k_range,
                       'mcv_loss': mcv_loss,
                       'rec_loss': rec_loss}
        return optimal_k, mcv_summary

    def plot_mcv_pca(mcv_summary: Dict[str, float], figdir: str, save_name: str, figsize= figsize):
        """
        Plots MCV loss of PCAs

        :param mcv_summary: Dictionary with MCV summary data
        :param figdir: path for which to save the figure
        :param save_name: name of file for figure
        :return:
        """
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 12})
        plt.plot(mcv_summary['k_range'], mcv_summary['mcv_loss'])

        idx = np.argmin(mcv_summary['mcv_loss'])
        optimal_k = mcv_summary['k_range'][idx]
        plt.scatter([optimal_k], [mcv_summary['mcv_loss'][idx]], c='k')
        plt.xlabel('Number of PCs')
        plt.ylabel("MCV Loss")
        plt.title("Optimal PCs = " + str(optimal_k))
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"{save_name}_pca_mcv.png"))
        #plt.close()
        plt.show()

        plt.rcParams.update({'font.size': 10})

    [optimal_k, mcv_summary]= mcv_calibrate_pca(mcv_data, selected=list(mcv_data.var_names[mcv_data.var[hvf_name]]), hvf_name=hvf_name)
    plot_mcv_pca(mcv_summary, figdir= savepath, save_name=iteration, figsize= figsize)
    return optimal_k

from typing import Union
from typing import List
# A function that creates Kamil's cellbrowser
def make_kamil_browser(adata: sc.AnnData,
                       browser_filepath: str,
                       browser_foldername: str,
                       browser_name: str = 'cell_browser',
                       which_meta: Union[str, List[str]] = 'all',
                       cluster_label: str = 'leiden_labels',
                       embedding: str = 'umap',
                       var_info: str = 'de_res',
                       var_list: List[str] = (
                               'auroc', 'log2Mean', 'log2Mean_other', 'log2FC', 'percentage', 'percentage_other',
                               'percentage_fold_change', 'mwu_U', 'mwu_pval', 'mwu_qval'),
                       auc_cutoff: int = 0.5,
                       pseudo_p_cutoff: int = None,
                       pseudo_fc_cutoff: int = None,
                       top_num_de: int = None,
                       health_field: str = "health",
                       donor_field: str = "Channel",
                       pval_precision: int = 3,
                       round_float: int = 2,
                       **kwargs):
    """
    Creates Kamil's cell browser

    :param adata: single-cell analysis data object
    :param browser_filepath: path for which to save the browser files
    :param browser_foldername: name for folder that stores the cell browser folder
    :param browser_name: name of file for cell browser
    :param which_meta: list of observations to include for cell browser
    :param cluster_label: variable name for cluster observation
    :param embedding: graph embedding
    :param var_info: name of variable for differential expression data
    :param var_list: list of variables to include for cell browser
    :param auc_cutoff: minimum cutoff for AUC
    :param pseudo_p_cutoff: maximum cutoff for pseudobulk p value
    :param pseudo_fc_cutoff: minimum cutoff for pseudobulk log fold change
    :param top_num_de: maximum number of top DE gene markers to include
    :param health_field: variable name for patient health observation (control vs. case)
    :param donor_field: variable name for patient ID observation (e.g. Channel, sample_id)
    :param pval_precision: precision for p values
    :param round_float: number of decimals to include when rounding floats
    :param kwargs:
    :return:
    """
    check_args(adata, browser_filepath, which_meta, cluster_label, embedding, var_info, var_list, auc_cutoff,
               pseudo_p_cutoff, pseudo_fc_cutoff)

    prepare_cb_files(adata, browser_filepath, which_meta, cluster_label, embedding, var_info, var_list, auc_cutoff,
                     pseudo_p_cutoff, pseudo_fc_cutoff, top_num_de, pval_precision, round_float)

    run_cbBuild(browser_filepath, browser_foldername, browser_name, health_field, donor_field, **kwargs)


def check_args(adata: sc.AnnData,
               browser_filepath: str,
               which_meta: str or iter,
               cluster_label: str,
               embedding: str,
               var_info: str,
               var_list: List[str],
               auc_cutoff: int,
               pseudo_p_cutoff: int,
               pseudo_fc_cutoff: int):
    """
    Checks the arguments and validates that data specified is available

    :param adata: single-cell analysis data object
    :param browser_filepath: path for which to save the browser files
    :param which_meta: list of observations to include for cell browser
    :param cluster_label: variable name for cluster observation
    :param embedding: graph embedding
    :param var_info: name of variable for differential expression data
    :param var_list: list of variables to include for cell browser
    :param auc_cutoff: minimum cutoff for AUC
    :param pseudo_p_cutoff: maximum cutoff for pseudobulk p value
    :param pseudo_fc_cutoff: minimum cutoff for pseudobulk log fold change
    :return:
    """

    # Make sure all of the vars are in the AnnData object
    all_vars = set([name.split(':')[1] for name in adata.varm[var_info].dtype.names])
    if all([i in adata.obs[cluster_label].cat.categories.values for i in all_vars]):
        warnings.warn('The following output is a list of cluster identifiers. You should to update Pegasus '
                      'to 1.0.0 or higher and rerun de_analysis. For now, it is fine.')
        # print(all_vars)
        all_vars = set([name.split(':')[0] for name in adata.varm[var_info].dtype.names])
    if not all(var in all_vars for var in var_list):
        raise ValueError(f'Not all {var_info} parameters are in data: {all_vars}')

    assert os.path.exists(browser_filepath)

    # Make sure embedding exists
    if f'X_{embedding}' not in adata.obsm:
        raise ValueError(f'{embedding} is not in the data')

    # Make sure cluster column exists
    if cluster_label not in adata.obs.columns:
        raise ValueError(f'{clustcol} is not in obs')

    # Lets make the metadata
    if not which_meta == 'all':
        if not set(which_meta).issubset(adata.obs.columns):
            raise ValueError('not all metadata in obs')

    if auc_cutoff:
        assert 'auroc' in all_vars
        assert 'auroc' in var_list

    if pseudo_p_cutoff or pseudo_fc_cutoff:
        assert 'pseudobulk_adj_p_val' in all_vars
        assert 'pseudobulk_adj_p_val' in var_list
        assert 'pseudobulk_log_fold_change' in all_vars
        assert 'pseudobulk_log_fold_change' in var_list


def prepare_cb_files(adata: sc.AnnData,
                     browser_filepath: str,
                     which_meta: str = 'all',
                     cluster_label: str = 'leiden_labels',
                     embedding: str = 'umap',
                     var_info: str = 'de_res',
                     var_list: List[str] = ('auroc', 'log2Mean', 'log2Mean_other', 'log2FC', 'percentage',
                                            'percentage_other', 'percentage_fold_change', 'mwu_U', 'mwu_pval',
                                            'mwu_qval'),
                     auc_cutoff: int = 0.5,
                     pseudo_p_cutoff: int = None,
                     pseudo_fc_cutoff: int = None,
                     top_num_de: int = None,
                     pval_precision: int = 3,
                     round_float: int = 2):
    """
    Prepare the cell browser input files

    :param adata: single-cell analysis data object
    :param browser_filepath: path for which to save the browser files
    :param browser_foldername: name for folder that stores the cell browser folder
    :param which_meta: list of observations to include for cell browser
    :param cluster_label: variable name for cluster observation
    :param embedding: graph embedding
    :param var_info: name of variable for differential expression data
    :param var_list: list of variables to include for cell browser
    :param auc_cutoff: minimum cutoff for AUC
    :param pseudo_p_cutoff: maximum cutoff for pseudobulk p value
    :param pseudo_fc_cutoff: minimum cutoff for pseudobulk log fold change
    :param top_num_de: maximum number of top DE gene markers to include
    :param pval_precision: precision for p values
    :param round_float: number of decimals to include when rounding floats
    :return:
    """

    # Lets make the cell browser files
    _make_expr_mtx(adata, browser_filepath)
    _make_cell_meta(adata, browser_filepath, cluster_label, which_meta)
    _make_embedding(adata, browser_filepath, embedding)
    _make_de_data(adata, browser_filepath, var_list, var_info, cluster_label, auc_cutoff, pseudo_p_cutoff,
                  pseudo_fc_cutoff, top_num_de, pval_precision, round_float)


def run_cbBuild(browser_filepath: str,
                browser_foldername: str,
                browser_name: str = 'cell_browser',
                health_field: str = "health",
                donor_field: str = "Channel",
                **kwargs):
    """
    Runs the UCSC cbBuild function, modifies output to fit standards of Kamil's web browser, and optionally re-.

    :param browser_filepath: path for which to save the browser files
    :param browser_name: name of file for cell browser
    :param health_field: variable name for patient health observation (control vs. case)
    :param donor_field: variable name for patient ID observation (e.g. Channel, sample_id)
    :param kwargs: To be passed to the _make_conf function
    :return:
    """

    # Lets make the conf file
    _make_conf(file_path=browser_filepath, name=browser_name, **kwargs)

    # Create the browser
    _make_browser(data_filepath=browser_filepath, browser_filepath=os.path.join(browser_filepath, browser_foldername),
                  health_field=health_field, donor_field=donor_field)


def _make_expr_mtx(adata: sc.AnnData, file_path: str):
    file_name = os.path.join(file_path, 'exprMatrix.tsv')

    try:
        import counts_to_csv as ctc
        ctc.counts_to_csv(adata, delimiter='tab', column_orient='obs-names', outfile=file_name)
        os.system(f'pigz -f -v {file_name}')
    except ModuleNotFoundError:
        print('Module counts_to_csv not found. Install for faster TSV creation!')
        print('https://github.com/swemeshy/counts_to_csv')
        # Kamil's write_tsv function from villani-lab/covid/make-cellbrowser.py
        if adata.X.getformat() == 'csr':
            x_matrix = adata.X.tocsc()
        else:
            x_matrix = adata.X
        f = open(file_name, 'w')
        head = ['gene'] + adata.obs.index.values.tolist()
        f.write('\t'.join(head))
        f.write('\n')
        for i in tqdm(range(x_matrix.shape[1])):
            f.write(adata.var.index.values[i])
            f.write('\t')
            row = x_matrix[:, i].todense()
            row.tofile(f, sep='\t', format='%.7g')
            f.write('\n')
        f.close()
        cmd = f'pigz -f -v {file_name}'
        os.system(cmd)



def _make_cell_meta(adata: sc.AnnData,
                    file_path,
                    cluster_label: str = 'leiden_labels',
                    which_meta: Union[str, List[str]] = 'all'):
    # Make the metadata
    if which_meta == 'all':
        cell_meta = adata.obs
    else:
        cell_meta = adata.obs[which_meta]

    # Need to make one of the columns the 'cluster' columns
    cell_meta = cell_meta.rename(columns={cluster_label: 'cluster'})

    # make cell_name column and make it first
    cell_meta['cellName'] = cell_meta.index
    cols = cell_meta.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cell_meta = cell_meta[cols]

    cell_meta.to_csv(os.path.join(file_path, 'meta_data.csv'), index=False)


def _make_embedding(adata: sc.AnnData,
                    file_path: str,
                    embedding: str = 'umap'):
    # Now lets make the embedding file
    embedding_file = pd.DataFrame(adata.obsm[f'X_{embedding}'], columns=['x', 'y'])
    embedding_file['cellName'] = adata.obs_names
    embedding_file = embedding_file[['cellName', 'x', 'y']]
    embedding_file.to_csv(os.path.join(file_path, 'embedding.csv'), index=False)


def _make_de_data(adata: sc.AnnData,
                  file_path: str,
                  var_list: List[str] = ('auroc', 'log2Mean', 'log2Mean_other', 'log2FC', 'percentage',
                                         'percentage_other', 'percentage_fold_change', 'mwu_U', 'mwu_pval', 'mwu_qval'),
                  var_info: str = 'de_res',
                  cluster_label: str = 'leiden_labels',
                  auc_cutoff: int = 0.5,
                  pseudo_p_cutoff: int = None,
                  pseudo_fc_cutoff: int = None,
                  top_num_de: int = None,
                  pval_precision: int = 3,
                  round_float: int = 2):

    # Get de columns and filter dataframe
    de_df = pd.DataFrame(columns=list(var_list) + ['gene', 'cluster'])
    for cluster in set(adata.obs[cluster_label]):
        df_dict = {}

        try:
            for var in var_list:
                df_dict[var] = adata.varm[var_info][f'{cluster}:{var}']
        except ValueError:
            for var in var_list:
                df_dict[var] = adata.varm[var_info][f'{var}:{cluster}']

        df = pd.DataFrame(df_dict)
        df['gene'] = adata.var_names
        df['cluster'] = cluster

        if any(cutoff is not None for cutoff in [auc_cutoff, pseudo_p_cutoff, pseudo_fc_cutoff]):
            if ('pseudobulk_adj_p_val' in var_list):
                df = df[(df['auroc'] > auc_cutoff) | ((df['pseudobulk_adj_p_val'] < pseudo_p_cutoff) &
                                                    (df['pseudobulk_log_fold_change'] > pseudo_fc_cutoff))]
            else:
                df = df[(df['auroc'] > auc_cutoff)]

        if top_num_de is not None:
            df = df.sort_values('auroc', ascending=False)[:top_num_de]

        # de_df = de_df.append(df, ignore_index=True)
        de_df= pd.concat([de_df, df], ignore_index=True)
    # Adjust so cluster and gene are the first two columns
    de_df = de_df[['cluster', 'gene'] + list(var_list)]

    # make p values display in scientific notation and round other float columns
    pval_labels = ['P_value', 'pval', 'pvalue', 'P.Value', 'adj.P.Val', 'p_val', 'pVal', 'Chisq_P', 'fdr', 'FDR']
    for col in var_list:
        if any([p in col for p in pval_labels]):
            de_df[col] = [np.format_float_scientific(num, precision=pval_precision) for num in de_df[col]]
        else:
            de_df[col] = de_df[col].astype(float).round(round_float)

    de_df.to_csv(os.path.join(file_path, 'de_data.csv'), index=False)


def _make_conf(file_path: str,
               name: str,
               priority: int = 10,
               tags: str = '10X',
               short_label: str = 'cell browser',
               expr_mtx: str = 'exprMatrix.tsv.gz',
               gene_id_type: str = 'auto',
               meta: str = 'meta_data.csv',
               enum_fields: str = 'cluster',
               coord_file: str = 'embedding.csv',
               coord_label: str = 'embedding',
               cluster_field: str = 'cluster',
               label_field: str = 'cluster',
               marker_file: str = 'de_data.csv',
               marker_label: str = '',
               show_labels: bool = True,
               radius: int = 5,
               alpha: float = 0.3,
               unit: str = 'log2_CPM',
               matrix_type: str = 'auto'):
    coord_dict = [{
        'file': f'{coord_file}',
        'flipY': False,
        'shortLabel': coord_label
    }]
    marker_dict = [
        {'file': f'{marker_file}', 'shortLabel': marker_label},
    ]
    tags = [tags]

    with open(os.path.join(file_path, 'cellbrowser.conf'), 'w') as f:
        f.write(
            f"name='{name}'\npriority={priority}\ntags={tags}\nshortLabel='{short_label}'\nexprMatrix='{expr_mtx}'\n"
            f"geneIdType='{gene_id_type}'\nmeta='{meta}'\nenumFields='{enum_fields}'\ncoords=\t{coord_dict}\n"
            f"clusterField='{cluster_field}'\nlabelField='{label_field}'\nmarkers=\t{marker_dict}\n"
            f"showLabels={show_labels}\nradius={radius}\nalpha={alpha}\nunit='{unit}'\nmatrixType='{matrix_type}'\n"
        )
        f.close()

def _swap_files(browser_filepath: str):
    import shutil
    shutil.copy2(os.path.join('/projects/home/adey01', 'cbfiles', 'index.html'), os.path.join(browser_filepath))
    for direc in ["css", "ext", "js"]:
        shutil.rmtree(os.path.join(browser_filepath, direc))
        shutil.copytree(os.path.join('/projects/home/adey01', 'cbfiles', direc), os.path.join(browser_filepath, direc))


def _make_browser(data_filepath: str, browser_filepath: str, health_field: str, donor_field: str):
    import sys

    # Need to add path environment variable for subprocess to work
    path = os.environ['PATH']

    # make sure conda path is included
    conda_path = '/'.join([sys.prefix, 'bin'])
    path = ':'.join([path, conda_path])

    print('Running cbBuild')
    completed_process = subprocess.run(['cbBuild', '-o', browser_filepath], cwd=data_filepath,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       env={'PATH': path})
    print(completed_process.stdout.decode())
    completed_process.check_returncode()

    # Change the files
    _swap_files(browser_filepath)

    # Change health and donor variables
    with open(os.path.join(browser_filepath, 'js', 'cellGuide.js'), 'r') as f:
        lines = f.readlines()
    lines[57] = f'var g_healthField = \"{health_field}\"\n'
    lines[58] = f'var g_donorField = \"{donor_field}\"\n'
    with open(os.path.join(browser_filepath, 'js', 'cellGuide.js'), 'w') as f:
        f.writelines(lines)

