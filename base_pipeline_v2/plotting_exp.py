from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import scanpy as sc
import os
import pandas as pd
#import seaborn as sns
#import sys
#sys.path.append("/projects/home/aantoinette")
from .utils_exp import balanced_sample
#from .utils import balanced_sample

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def plot_obs_feature_on_umap_separately(data, feature, save_file, size=None):
    # Validate inputs
    assert hasattr(data, 'obs'), "Input data must be an AnnData object."
    assert feature in data.obs.columns, f"Feature '{feature}' not found in data.obs."
    assert isinstance(save_file, str), "save_file must be a valid string path."

    # Ensure UMAP is available in data.obsm
    assert 'X_umap' in data.obsm, "UMAP coordinates ('X_umap') not found in data.obsm."

    # Check if the feature has unique values
    unique_values = data.obs[feature].unique()
    if len(unique_values) == 0:
        raise ValueError(f"Feature '{feature}' has no unique values. Unable to plot.")

    # Calculate number of rows for subplots based on unique feature values
    n_rows = int(np.ceil(len(unique_values) / 4.0))

    # Set up subplots for each unique feature value
    fig, axes = plt.subplots(n_rows, 4, figsize=(10, n_rows * 3))
    axes = axes.flatten()

    plt.rcParams.update({'axes.labelsize': 0})  # Hide axis labels

    # Plot UMAP for each donor (feature value)
    if len(unique_values) < 20:
        for donor, ax in zip(unique_values, axes):
            sc.pl.umap(data, color=[feature], groups=donor, ax=ax, show=False,
                       legend_loc="none", size=size)
            ax.xaxis.label.set_size(0)
            ax.yaxis.label.set_size(0)
            ax.set_title(donor, fontsize=22)

        # Remove any empty axes that don't have a title
        for ax in axes:
            if not ax.get_title():
                fig.delaxes(ax)

        plt.tight_layout()

        # Save the figure
        fig.savefig(save_file, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        print(f"Feature '{feature}' has more than 20 unique values. Skipping individual plots.")


from matplotlib import pyplot as plt, cm
#from mpl_table import Table, ColumnDefinition  # assuming mpl_table is installed
import pandas as pd
import numpy as np
import os
from plottable.cmap import normed_cmap
from matplotlib import colormaps
from plottable import ColumnDefinition, Table


def plot_calc_metrics(metrics_list, index_names, savepath=None, min_max_scale=True):
    # Validate inputs
    if not metrics_list:
        raise ValueError("metrics_list is empty. Please provide valid metrics.")

    if len(metrics_list) != len(index_names):
        raise ValueError("Length of metrics_list must match length of index_names")

    # Create DataFrame from metrics and add index names as 'Method' column
    data = pd.DataFrame(metrics_list, index=index_names)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Method'}, inplace=True)

    # Define the desired column order
    ordered_columns = ['Method', 'ASW_l', 'clisi', 'ARI', 'NMI', 'kSIM', 'ASW_b', 'ilisi', 'kBet']

    # Ensure the DataFrame contains all expected columns
    missing_columns = [col for col in ordered_columns[1:] if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in metrics data: {', '.join(missing_columns)}")

    data = data[ordered_columns]

    # Normalize if required
    if min_max_scale:
        global_min, global_max = data.iloc[:, 1:].min().min(), data.iloc[:, 1:].max().max()  # Exclude 'Method'
        data.iloc[:, 1:] = (data.iloc[:, 1:] - global_min) / (global_max - global_min)

    # Define column aesthetics for the table
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"})
    ]

    cmap = colormaps.get_cmap('PRGn')
    for col in ordered_columns[1:]:
        column_definitions.append(
            ColumnDefinition(
                col,
                title=col,
                width=1,
                textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.25}},
                cmap=cmap,
                formatter="{:.2f}",
            )
        )

    # Create the figure for plotting
    fig, ax = plt.subplots(figsize=(len(data.columns) * 1.25, len(data) * 0.5))
    table_plot = Table(
        data,
        cell_kw={"linewidth": 0, "edgecolor": "k"},
        column_definitions=column_definitions,
        ax=ax,
        row_dividers=True,
        footer_divider=True,
        textprops={"fontsize": 10, "ha": "center"},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
        index_col="Method",
    )

    # Annotate the plot to visually separate the two categories
    ax.annotate('Bio Conservation', xy=(0.27, 1.0), xycoords='axes fraction', ha='center', fontsize=12,
                color='black', weight='bold')
    ax.annotate('Batch Correction', xy=(0.80, 1.0), xycoords='axes fraction', ha='center', fontsize=12,
                color='black', weight='bold')

    # Remove axis for a cleaner table plot
    ax.axis('off')

    # Save the plot if savepath is provided
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        save_file = os.path.join(savepath, 'metrics_aesthetic_plot.png')
        fig.savefig(save_file, bbox_inches='tight', dpi=300)

    # Display the plot
    plt.show()
    plt.close()


def compare_clusters(adata1, adata2, cluster_label1, cluster_label2, label1_name, label2_name, savepath=None):
    """
    Compare clustering results between two AnnData objects and generate a contingency table heatmap.

    Parameters:
    - adata1, adata2: AnnData objects containing the clustering results
    - cluster_label1, cluster_label2: Column names in `obs` for cluster assignments in each AnnData
    - label1_name, label2_name: Intuitive names for the clusters (e.g., "scVI", "Harmony")
    - save_plot (bool): Whether to save the plot (default is False)
    - savepath (str or None): Path to save the plot, required if save_plot is True

    Returns:
    - contingency_table: The contingency table (DataFrame)
    """

    # Extract cluster labels
    cluster1 = adata1.obs[cluster_label1]
    cluster2 = adata2.obs[cluster_label2]

    # Make sure both have the same number of cells (matching cell count)
    assert len(cluster1) == len(cluster2), "The two datasets must have the same number of cells!"

    # Create a DataFrame of the pairwise comparison
    comparison_df = pd.DataFrame({
        f'{label1_name}_cluster': cluster1,
        f'{label2_name}_cluster': cluster2
    })

    # Create a contingency table (cross-tabulation) to compare the clusters
    contingency_table = pd.crosstab(comparison_df[f'{label1_name}_cluster'], comparison_df[f'{label2_name}_cluster'])

    # Print the contingency table
    #print("Contingency Table:\n", contingency_table)

    import seaborn as sns

    # Generate the heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.title(f"Contingency Table: {label1_name} vs {label2_name}", fontsize=14)
    plt.xlabel(f"Clusters from {label2_name}", fontsize=14)
    plt.ylabel(f"Clusters from {label1_name}", fontsize=14)

    # Save the plot if requested
    #if save_plot:
    if savepath:
        plt.savefig(savepath)
        print(f"Plot saved to {savepath}")
        #else:
            #print("Please provide a save path when save_plot is True.")

    plt.show()

    return contingency_table




def plot_umap(adata, col, label, figsize, legend_loc, savepath=None):
    plt.figure(figsize=figsize)
    # Create dot plot without immediately displaying it
    d = sc.pl.umap(adata, color=[col], palette=sc.pl.palettes.vega_20_scanpy, show=False,
                   legend_loc=legend_loc)
    # Adjust layout to avoid cutting off
    plt.tight_layout()
    # Save plot with tight bounding box if savepath is provided
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        save_file = os.path.join(savepath, f'umap_{label}.png')
        plt.savefig(save_file, bbox_inches='tight')  # Use plt.savefig instead of fig.savefig

    # Show the plot
    plt.show()


def visualize_leiden_umap_compo_dotplots(adata, max_cells_plot, leiden_res_list, save_dir,
                                   generate_GeneralMarkers_dotplot_per_leiden_cluster, list_of_variables_to_check_clusters, general_markers):
    clusters_folder = os.path.join(save_dir, "clusters/")
    os.makedirs(clusters_folder, exist_ok=True)

    for res in leiden_res_list:
        leiden_cluster = "leiden_res_" + str(res)
        res_folder = os.path.join(clusters_folder, leiden_cluster)
        os.makedirs(res_folder, exist_ok=True)
        # UMAP plot
        # Subsetting for optimal plotting
        fraction = min(1, max_cells_plot / adata.shape[0])
        random_indices = balanced_sample(adata.obs, cols=leiden_cluster, frac=fraction, shuffle=True,
                                         random_state=42).CellID
        # sc.set_figure_params(figsize=(7, 5))
        fig = sc.pl.embedding(adata[random_indices, :], basis="X_umap",
                              color=[leiden_cluster], vmin="p.5", vmax="p99",
                              title=leiden_cluster, show=True, return_fig=True, size=10, legend_loc="on data")
        plt.savefig(os.path.join(res_folder, f"UMAP_clusters_res{res}.pdf"), bbox_inches='tight', pad_inches=0, dpi=300)

        # BarPlots
        pdf_pages = PdfPages(os.path.join(res_folder, f"Barplots_covariates_across_clusters_{res}.pdf"))
        for variable in list_of_variables_to_check_clusters:
            sc.set_figure_params(figsize=(15, 5))
            composition_barplot(adata, xattr=leiden_cluster, yattr=variable, title=f"Distribution of {variable}",
                                save_pdf=pdf_pages)
        pdf_pages.close()

        # Visualize general markers
        if generate_GeneralMarkers_dotplot_per_leiden_cluster:
            print(f"****** Generating dotplot for general markers at resolution {res} ...")
            fig_dotplot = sc.pl.dotplot(adata[random_indices, :], var_names=general_markers, groupby=leiden_cluster,
                                        standard_scale='var', use_raw=False, dendrogram=False, show=False,
                                        return_fig=True).add_totals().style(dot_edge_color='black', dot_edge_lw=0.25,
                                                                            cmap="Reds").savefig(
                os.path.join(res_folder, f"DotPlot_GeneralMarkers_in_{leiden_cluster}.pdf"),
                bbox_inches='tight', pad_inches=0, dpi=300)



def composition_barplot(adata, xattr, yattr, title, save_pdf, color_dict=None, fig_size=None):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.crosstab(adata.obs.loc[:, xattr], adata.obs.loc[:, yattr])
    df = df.div(df.sum(axis=1), axis=0) * 100.0
    if (color_dict is None):
        ax = df.plot(kind="bar", stacked=True, figsize=fig_size, legend=True, grid=False)
    else:
        ax = df.plot(kind="bar", stacked=True, figsize=fig_size, legend=True, grid=False, color=color_dict)
    ax.figure.subplots_adjust(right=0.9)
    for i in range(len(list(df.index))):
        x_feature = list(df.index)[i]
        feature_count = adata[adata.obs[xattr] == x_feature, :].shape[0]
        ax.annotate(str(feature_count), (ax.patches[i].get_x(), 101))
    ax.annotate("Cell no.", ((ax.patches[i].get_x() + 1), 101), annotation_clip=False)
    ax.set_title(title, fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    if save_pdf != None:
        save_pdf.savefig(ax.figure, bbox_inches="tight")




def visualize_general_markers(adata, max_cells_plot, col,
                        general_markers, category, save_dir, save_output=None):

    fraction = min(1, max_cells_plot / adata.shape[0])
    random_indices = balanced_sample(adata.obs, cols=col, frac=fraction, shuffle=True,
                                     random_state=42).CellID
    print(f"****** Visualizing general markers UMAP ...")
    vizMarkers_folder = os.path.join(save_dir, "visualize_markers/")
    os.makedirs(vizMarkers_folder, exist_ok=True)
    gMarkers = os.path.join(vizMarkers_folder, "general_markers/")
    os.makedirs(gMarkers, exist_ok=True)


    fig = sc.pl.embedding(adata[random_indices, :], basis='X_umap', vmin='p1', vmax='p99',
                          color=general_markers,
                          wspace=0.4, ncols=4, return_fig=True)
    plt.savefig(gMarkers + "/General_markers.pdf", bbox_inches='tight', pad_inches=0, dpi=300)

    if save_output:
        # Save the final adata object as .h5ad
        output_folder = os.path.join(save_dir, "output")
        os.makedirs(output_folder, exist_ok=True)
        adata_h5ad_path = os.path.join(output_folder, f"{category}_processed.h5ad")
        adata.write(adata_h5ad_path)
        print(f"Final adata object saved to {adata_h5ad_path}")
        print("Processing complete.")
    return adata



def create_hexbin_plot(data, dataset_name='',group='', savepath='', axlines=[], col_wrap=1):
    #print('hello')
    import seaborn as sns
    # fig, ax = plt.subplots(figsize=(4*len(set(data.obs[group])),4))
    g = sns.FacetGrid(data.obs, col=group, col_wrap=col_wrap,
                      height=4, aspect=1)
    g.map(plt.hexbin, 'n_genes', 'percent_mito', bins='log',
          gridsize=50, xscale='log', edgecolors='none', linewidths=0)
    for ax in g.axes.flat:
        ax.axhline(y=axlines[1], color='red', linestyle='--')
        ax.axvline(x=axlines[0], color='red', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.xaxis.set_tick_params(labelbottom=True)

        # Add axis labels and a title
    g.set_axis_labels('Number of Genes', 'Percent Mito')
    # g.fig.suptitle('Hexbin plot of gene count vs percent mitochondrial reads by sample', wrap=True)
    plt.tight_layout()
    # Save the figure
    plt.savefig(savepath+'/'+dataset_name+ group+'_hexbin_plot.png')
    plt.close()

