import os
import scanpy as sc
#import sys
#sys.path.append("/projects/home/aantoinette")
from .utils_exp import run_marker_genes_analysis

def compute_leiden_clusters(adata, list_of_variables_to_check_clusters, leiden_res_list, save_dir, other_vars,
                             neighbors_key):
    # Check if the input lists are not empty
    assert isinstance(list_of_variables_to_check_clusters, list) and len(list_of_variables_to_check_clusters) > 0, \
        "list_of_variables_to_check_clusters must be a non-empty list"
    assert isinstance(leiden_res_list, list) and len(leiden_res_list) > 0, \
        "leiden_res_list must be a non-empty list"
    assert isinstance(other_vars, list), "other_vars must be a list"

    # Check if UMAP embedding exists in the adata object
    assert 'X_umap' in adata.obsm, "UMAP coordinates (X_umap) not found in adata.obsm"

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize figure for UMAP plot
    fig = sc.pl.embedding(adata, basis='X_umap', vmin='p1', vmax='p99',
                          color=list_of_variables_to_check_clusters + other_vars,
                          wspace=0.4, ncols=3, return_fig=True)

    # Create a folder to save clusters
    clusters_folder = os.path.join(save_dir, "clusters/")
    os.makedirs(clusters_folder, exist_ok=True)

    print("****** Computing clusters for resolutions specified...")

    # Iterate over the list of resolutions
    for res in leiden_res_list:
        leiden_cluster = "leiden_res_" + str(res)
        res_folder = os.path.join(clusters_folder, leiden_cluster)
        os.makedirs(res_folder, exist_ok=True)

        # Compute Leiden clusters for the given resolution
        print(f"--- Computing clusters at resolution {res}...")

        # Check if the leiden cluster already exists
        if leiden_cluster in adata.obs:
            print(f"Warning: {leiden_cluster} already exists in adata.obs, overwriting...")

        # Compute leiden clustering
        sc.tl.leiden(adata, resolution=res, key_added=leiden_cluster, neighbors_key=neighbors_key)

        # Ensure the cluster labels are integers
        adata.obs[leiden_cluster] = adata.obs[leiden_cluster].astype(int)

        # Save the cluster data to a CSV file
        clusters_csv_path = os.path.join(res_folder, f'clusters_{leiden_cluster}.csv')
        adata.obs.to_csv(clusters_csv_path, index=True)
        print(f"Cluster data saved to {clusters_csv_path}")

        # Optionally, convert the cluster labels to strings for further processing
        adata.obs[leiden_cluster] = adata.obs[leiden_cluster].astype(str)

    print("****** Cluster computation completed.")


def calc_markers(adata, leiden_res_list, save_dir, compute_markers, save_excel=None, visualize_markers=True):
    # Validate input types
    assert isinstance(leiden_res_list, list) and len(leiden_res_list) > 0, "leiden_res_list must be a non-empty list"
    assert isinstance(save_dir, str) and os.path.isdir(save_dir), "save_dir must be a valid directory path"
    assert isinstance(compute_markers, bool), "compute_markers must be a boolean value"
    assert isinstance(visualize_markers, bool), "visualize_markers must be a boolean value"

    # Ensure the clusters directory exists
    clusters_folder = os.path.join(save_dir, "clusters/")
    os.makedirs(clusters_folder, exist_ok=True)

    # Iterate over each resolution in leiden_res_list
    for res in leiden_res_list:
        print(f"Resolution: {res}")
        leiden_cluster = "leiden_res_" + str(res)
        print(f"Leiden cluster: {leiden_cluster}")

        # Ensure the leiden_cluster exists in adata.obs
        if leiden_cluster not in adata.obs:
            print(f"*** {leiden_cluster} not found in adata.obs. Skipping this resolution. ***")
            continue

        # Create a folder for this resolution
        res_folder = os.path.join(clusters_folder, leiden_cluster)
        os.makedirs(res_folder, exist_ok=True)

        if compute_markers:
            # Skip if there is only one cluster in the resolution
            if len(set(adata.obs[leiden_cluster])) == 1:
                print(f"*** Only one cluster in resolution {res}. Skipping marker computation. ***")
            else:
                print(f"--- Computing markers for {leiden_cluster}...")

                # Create markers folder for this resolution
                markers_folder = os.path.join(res_folder, "markers/")
                os.makedirs(markers_folder, exist_ok=True)

                # Define the key for differential expression results
                de_key = "de_res_" + str(res)

                # Handle the default save_excel and visualize_markers arguments
                save_excel = save_excel if save_excel is not None else False
                visualize_markers = visualize_markers if visualize_markers is not None else True

                # Run the marker genes analysis function
                run_marker_genes_analysis(
                    adata, de_key=de_key, save_dir=markers_folder,
                    leiden_cluster=leiden_cluster, logFC_thr=0.25, pval_thr=0.05,
                    pct_nz_thr=0.25, save_excel=save_excel, visualize_markers=visualize_markers
                )

    print("Marker calculation completed.")
