import scanpy as sc
from harmony import harmonize
from scanpy.external.pp import scanorama_integrate
from scanpy.external.pp import bbknn

##########################################

# import scanpy as sc
# from harmony import harmonize
# from bbknn import bbknn
# from scanorama import integrate_scanorama as scanorama_integrate

class IntegrationPipeline:
    def __init__(self, adata, method, **kwargs):
        """
        Initialize the integration pipeline with the chosen method and parameters.

        Parameters:
            adata (AnnData): The annotated data matrix.
            method (str): The integration method to use. Options are 'harmony', 'bbknn', 'scanorama', or 'scvi'.
            **kwargs: Additional parameters specific to each integration method.
        """
        self.adata = adata
        self.method = method
        self.params = kwargs

    def integrate(self):
        if self.method == 'harmony':
            return self._harmony_fn()
        elif self.method == 'bbknn':
            return self._bbknn_fn()
        elif self.method == 'scanorama':
            return self._scanorama_fn()
        elif self.method == 'scvi':
            return self._scvi_fn()
        else:
            raise ValueError(f"Integration method '{self.method}' is not recognized.")

    def _harmony_fn(self):
        batch_key = self.params.get("batch_key", "batch")
        n_neighbors = self.params.get("n_neighbors", 15)
        optimal_pcs = self.params.get("optimal_pcs", 30)
        hvg = self.params.get("hvg", "highly_variable")

        print("****** Performing PCA...")
        sc.pp.pca(self.adata, n_comps=optimal_pcs, use_highly_variable=hvg, zero_center=False)

        print("****** Running Harmony integration...")
        rep = "pca_harmony"
        self.adata.obsm["X_" + rep] = harmonize(
            self.adata.obsm['X_pca'], self.adata.obs, batch_key=batch_key,
            random_state=25, n_jobs=5, max_iter_harmony=100
        )

        print("****** Finding neighbors...")
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=optimal_pcs, use_rep='X_pca_harmony', key_added='harmony')

        print("****** Computing UMAP...")
        sc.tl.umap(self.adata, neighbors_key='harmony')
        return self.adata

    def _bbknn_fn(self):
        batch_key = self.params.get("batch_key", "batch")
        n_neighbors = self.params.get("n_neighbors", 15)
        optimal_pcs = self.params.get("optimal_pcs", 30)
        hvg = self.params.get("hvg", "highly_variable")

        print("****** Performing PCA...")
        sc.pp.pca(self.adata, n_comps=optimal_pcs, use_highly_variable=hvg, zero_center=False)

        print("****** Running Bbknn integration...")
        bbknn(self.adata, batch_key=batch_key, use_rep='X_pca', n_pcs=optimal_pcs, key_added='bbknn')

        print("****** Computing UMAP...")
        sc.tl.umap(self.adata, neighbors_key='bbknn')
        return self.adata

    def _scanorama_fn(self):
        batch_key = self.params.get("batch_key", "batch")
        n_neighbors = self.params.get("n_neighbors", 15)
        optimal_pcs = self.params.get("optimal_pcs", 30)
        hvg = self.params.get("hvg", "highly_variable")

        print("****** Performing PCA...")
        sc.pp.pca(self.adata, n_comps=optimal_pcs, use_highly_variable=hvg, zero_center=False)

        print("****** Running Scanorama integration...")
        idx = self.adata.obs.sort_values(batch_key).index
        adata_new = self.adata[idx, :]

        scanorama_integrate(adata_new, key=batch_key, basis='X_pca', adjusted_basis='X_scanorama')

        print("****** Finding neighbors...")
        sc.pp.neighbors(adata_new, n_neighbors=n_neighbors, n_pcs=optimal_pcs, use_rep='X_scanorama', key_added='scanorama')

        print("****** Computing UMAP...")
        sc.tl.umap(adata_new, neighbors_key='scanorama')
        return adata_new

    def _scvi_fn(self):
        savepath = self.params.get("savepath", "./scvi_output")
        epochs = self.params.get("epochs", 400)
        disc_vars = self.params.get("discrete_variables", ['Batch'])
        cont_vars = self.params.get("continuous_variables", [])
        n_layers = self.params.get("n_layers", 2)
        n_latent = self.params.get("n_latent", 30)
        username_on_dgx = self.params.get("username_on_dgx",'aantoinette')
        working_directory_dgx = self.params.get("working_directory_dgx",'/home/aantoinette/endocrinopathies')



        print("****** Running scVI integration...")
        import run_scvi_on_dgx as rsod
        rsod.run_scvi_on_dgx(
            self.adata, discrete_variables=disc_vars,
            continous_variables=cont_vars,
            working_directory_mega=savepath,
            working_directory_dgx=working_directory_dgx,
            username_on_dgx=username_on_dgx, tmux_session_name='1', no_epochs=epochs,
            n_layers=n_layers, n_latent = n_latent
        )

        print("****** Loading scVI results...")
        adata_scvi = sc.read(f"{savepath}/after_scVI_training.h5ad")
        return adata_scvi

#Example Usage
#python
#Copy code
# Assuming `adata` is your AnnData object

#
# #################
# # Using Harmony integration
# harmony_pipeline = IntegrationPipeline(
#     adata, method="harmony", batch_key="Batch", n_neighbors=20, optimal_pcs=22, hvg='highly_variable'
# )
# adata_harmony = harmony_pipeline.integrate()
#
# # Using BBKNN integration
# bbknn_pipeline = IntegrationPipeline(
#     adata, method="bbknn", batch_key="Batch", n_neighbors=15, optimal_pcs=22, hvg='highly_variable'
# )
# adata_bbknn = bbknn_pipeline.integrate()
#
# # Using Scanorama integration
# scanorama_pipeline = IntegrationPipeline(
#     adata, method="scanorama", batch_key="Batch", n_neighbors=10, optimal_pcs=22, hvg='highly_variable'
# )
# adata_scanorama = scanorama_pipeline.integrate()
#
# # Using scVI integration
# scvi_pipeline = IntegrationPipeline(
#     adata, method="scvi", savepath='/projects/home/aantoinette/endocrinopathies/integration_101424/',
#     epochs=30, discrete_variables=['Batch'], continuous_variables=[], n_layers=2, n_latent=30
# )
# adata_scvi = scvi_pipeline.integrate()



