#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /GeneSPOT_rep/src/preprocess.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2025-08-05 02:18:36
-------------------------------------------------
Change Activity :
  LastEditTime  : 2025-08-05 02:18:36
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import scipy
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from anndata import AnnData


def preprocess_adata(
    adata: AnnData,
    norm: bool = True,
    target_sum: float = 1e4, 
    base: int = 2,
    zero_center: bool = True,
    regressout: bool = False,
    scale: bool = True,
    use_csr: bool = True,
) -> AnnData:
    adata = adata.copy()
    
    if 'raw' in adata.layers:
        if scipy.sparse.issparse(adata.layers['raw']):
            adata.layers['raw'] = adata.layers['raw'].toarray().copy()
        adata.X = adata.layers['raw'].copy()
    else:
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray().copy()
        adata.layers['raw'] = adata.X.copy()
    
    if norm:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        if use_csr:
            adata.layers['norm'] = csr_matrix(adata.X.copy())
        else:
            adata.layers['norm'] = adata.X.copy()

    sc.pp.log1p(adata, base=base)
    layer_name = f'log1p{"_norm" if norm else ""}'
    if use_csr:
        adata.layers[layer_name] = csr_matrix(adata.X.copy())
    else:
        adata.layers[layer_name] = adata.X.copy()

    if scale:
        sc.pp.scale(adata, zero_center=zero_center)
    
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray().copy()

    scaled_layer_name = f'log1p{"_norm" if norm else ""}_scaled'
    if use_csr:
        adata.layers[scaled_layer_name] = csr_matrix(adata.X.copy())
    else:
        adata.layers[scaled_layer_name] = adata.X.copy()
    
    if regressout:
        adata.obs['total_counts'] = adata.layers['raw'].sum(axis=1)
        sc.pp.regress_out(adata, ['total_counts'])
        if use_csr:
            adata.layers[scaled_layer_name] = csr_matrix(adata.X.copy())
        else:
            adata.layers[scaled_layer_name] = adata.X.copy()
    
    return adata



def filter_features(
        adata: sc.AnnData,
        layer: str = None,
        flavor: str = 'seurat_v3',
        expr_percent_filter: bool = False,
        min_expr_percent: float = 0.01,
        max_expr_percent: float = 0.5,
        n_variable_genes: int = 2000,
) -> np.ndarray:
    """
    Narrow down the gene list for gene-gene distance computation by focusing on the top
    2000 variable genes expressed by 1% - 50% of cells.

    :param adata: a scanpy Anndata object
    :param layer: the layer with count data (e.g. 'counts', which can be created
           as `adata.layers['counts']=adata.raw.X.copy()`)
    :param min_expr_percent: minimum fraction of cells expressing the gene
    :param max_expr_percent: maximum fraction of cells expressing the gene
    :param n_variable_genes: number of variable genes to use (default: 2000)
    :return: the cell-cell graph distance matrix
    """
    adata = adata.copy()
    if layer is not None and layer not in adata.layers.keys():
        raise ValueError(f'Layer {layer} not found in adata. Available {list(adata.layers.keys())}')
    if adata.n_vars < n_variable_genes:
        raise ValueError(f'This method should be run with at least 2000 features (genes). The data has {adata.n_vars}')

    sc.pp.calculate_qc_metrics(adata, layer=layer, inplace=True)
    sc.pp.highly_variable_genes(adata, layer=layer, n_top_genes=n_variable_genes, flavor=flavor, inplace=True)
    if expr_percent_filter:
        expr_percent = adata.var['n_cells_by_counts'] / adata.n_obs
        genes = adata.var_names[adata.var['highly_variable'] &
                                (expr_percent > min_expr_percent) & (expr_percent < max_expr_percent)]
    else:
        genes = adata.var_names[adata.var['highly_variable']]
    genes_sorted = adata[:, genes].var.sort_values('variances_norm', ascending=False).index
    return genes_sorted.values



def concat_features(adata_list, feature_groups=None):
    xs = []
    xs_raw = []
    var_names = []

    if feature_groups is not None:
        feature_types = []
    for i, adata in enumerate(adata_list):
        x = adata.X.toarray().copy() if issparse(adata.X) else adata.X.copy()
        raw_x = adata.layers['raw'].toarray().copy() if issparse(adata.layers['raw']) else adata.layers['raw'].copy()
        xs.append(x)
        xs_raw.append(raw_x)
        var_names.extend(adata.var_names.tolist())
        if feature_groups is not None:
            feature_types.extend([feature_groups[i]] * adata.shape[1])

    x = np.concatenate(xs, axis=1)
    x_raw = np.concatenate(xs_raw, axis=1)

    adata_concated = sc.AnnData(
        x,
        obs=adata.obs, 
        obsm=adata.obsm,
        layers={'raw': x_raw},
    )

    adata_concated.var_names = var_names
    adata_concated.obs_names = adata.obs_names
    
    if feature_groups is not None:
        adata_concated.var['feature_types'] = feature_types
    
    return adata_concated

