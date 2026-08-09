#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /GeneSPOT/genespot/genespot.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2025-08-04 11:19:48
-------------------------------------------------
Change Activity :
  LastEditTime  : 2025-08-04 11:19:48
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


from typing import Optional
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity


def _add_landmarks(
    adata: sc.AnnData,
    landmark_key: str,
    landmark_subset: list = None,
    layer: str = None,
    landmark_expression_value: float = 1.0,
    non_landmark_expression_value: float = 0.0,
    landmark_prefix: str = "SpatialLandmark",
    add_background: bool = False,
    background_type: str = 'uniform',
    background_name: Optional[str] = None,
    background_prefix: str = "SpatialBackground",
    n_perm: int = 10,
):
    """
    Add landmark features and optionally background features to AnnData object.
    
    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to add features to.
    landmark_key : str
        Column name in adata.obs containing landmark labels.
    landmark_subset : list, optional
        Subset of landmarks to add. If None, uses all categories in landmark_key.
    layer : str, optional
        Layer to use for feature expression. If None, uses adata.X.
    landmark_expression_value : float, default 1.0
        Value to assign when cell belongs to landmark.
    non_landmark_expression_value : float, default 0.0
        Value to assign when cell does not belong to landmark.
    landmark_prefix : str, default 'SpatialLandmark'
        Prefix for landmark feature names.
    add_background : bool, default False
        Whether to add a background feature.
    background_type : str, default 'uniform'
        Type of background expression to add:
        - 'uniform': All values set to 1
        - 'mean': Mean expression across all genes for each cell
        - 'sum': Sum of expression across all genes for each cell
    background_name : str, default 'Background'
        Name for the background feature.
    background_prefix : str, default 'SpatialBackground'
        Prefix for background feature type.
    
    Returns
    -------
    sc.AnnData
        AnnData object with landmark and optional background features added.
    """
    adata = adata.copy()

    if landmark_key not in adata.obs.columns:
        raise ValueError(f"Column '{landmark_key}' does not exist in adata.obs")
    if 'feature_names' not in adata.var.columns:
        raise ValueError("adata.var does not have 'feature_names' column")
    if 'feature_types' not in adata.var.columns:
        raise ValueError("adata.var does not have 'feature_types' column")
    if 'feature_annotations' not in adata.var.columns:
        adata.var['feature_annotations'] = adata.var['feature_types']

    
    if not landmark_subset:
        landmark_subset = adata.obs[landmark_key].cat.categories.tolist()

    # Initialize landmark features DataFrame with default values
    landmark_exp_profiles = pd.DataFrame(
        non_landmark_expression_value,
        index=adata.obs_names,
        columns=[f"{landmark_prefix}_{landmark}" for landmark in landmark_subset]
    )
    landmark_labels = adata.obs[landmark_key]

    for landmark in landmark_subset:
        matching_cells = landmark_labels[landmark_labels == landmark].index.tolist()
        if matching_cells:
            landmark_exp_profiles.loc[matching_cells, f'{landmark_prefix}_{landmark}'] = landmark_expression_value
        else:
            print(f"Warning: landmark '{landmark}' has no valid assigned cells")

    landmark_adata = sc.AnnData(X=landmark_exp_profiles)
    landmark_adata.var['feature_names'] = landmark_exp_profiles.columns
    landmark_adata.var['feature_types'] = landmark_prefix
    landmark_adata.var['feature_annotations'] = landmark_exp_profiles.columns

    data = adata.layers[layer] if layer is not None else adata.X
    if scipy.sparse.issparse(data):
        data = data.toarray()
    
    # Prepare list of feature matrices to concatenate
    feature_matrices = [data, landmark_exp_profiles.values]
    var_dfs = [adata.var, landmark_adata.var]
    
    if add_background:
        background_gexp = []
        if n_perm > 0:
            for r in range(n_perm):
                if background_type == 'sum':
                    background_values = np.random.permutation(data.sum(axis=1))
                elif background_type == 'mean':
                    background_values = np.random.permutation(data.mean(axis=1))
                elif background_type == 'uniform':
                    background_values = np.random.uniform(0, 1, size=data.shape[0])
                background_gexp.append(background_values)
            background_gexp = np.stack(background_gexp, axis=1)
            background_feature_name = [f"{background_prefix}_{background_type}_{r}" for r in range(n_perm)]
            background_feature_annotations = [f'{background_prefix}' for r in range(n_perm)]
        
        elif n_perm == 0:
            if background_type == 'sum':
                background_values = data.sum(axis=1)
            elif background_type == 'mean':
                background_values = data.mean(axis=1)
            elif background_type == 'uniform':
                background_values = np.random.uniform(0, 1, size=data.shape[0])
            elif background_type == 'ones':
                background_values = np.ones(data.shape[0])

            background_gexp = background_values.reshape(-1, 1)
            background_feature_name = [f"{background_prefix}_{background_type}"]
            background_feature_annotations = [f'{background_prefix}']

        background_exp_profile = pd.DataFrame(
            background_gexp,
            index=adata.obs_names,
            columns=background_feature_name
        )     
        background_var = pd.DataFrame({
            'feature_names': background_feature_name,
            'feature_types': background_prefix,
            'feature_annotations': background_feature_annotations
        }, index=background_feature_name)
        var_dfs.append(background_var)
        feature_matrices.append(background_exp_profile.values)
    
    # Concatenate all features
    adata_with_features = sc.AnnData(
        X=np.concatenate(feature_matrices, axis=1),
        obs=adata.obs
    )
    adata_with_features.var = pd.concat(var_dfs, axis=0)
    adata_with_features.obsm = adata.obsm.copy()
    adata_with_features.obsp = adata.obsp.copy()
    adata_with_features.uns = adata.uns.copy()

    
    return adata_with_features


def get_coarsen_geodesic_distance(adata, method='paga_guided_umap', connect_threshold=0.5): 
    if method == 'paga_graph':
        adj_mat_use = adata.uns['paga']['connectivities'].toarray()
        adj_mat_use = 1 / adj_mat_use

        np.fill_diagonal(adj_mat_use, 0)
        G = nx.from_numpy_array(adj_mat_use)

        len_path = dict(nx.all_pairs_dijkstra_path_length(G))
        ground_distance_mat = np.inf*np.ones(shape=(len(len_path), len(len_path)))

        for i in range(ground_distance_mat.shape[0]):
            for j in range(ground_distance_mat.shape[1]):
                ground_distance_mat[i,j] = len_path[i][j]
        np.fill_diagonal(ground_distance_mat, 0)
        
    elif method =='paga':
        adj_mat_use = adata.uns['paga']['connectivities'].toarray()
        adj_mat_use = 1 - adj_mat_use
        np.fill_diagonal(adj_mat_use, 0)
        ground_distance_mat = adj_mat_use
        
    elif method =='euclidean':   #lower bound of MED(approximation)
        ground_distance_mat = squareform(pdist(adata.uns['paga']['pos']))
        
    elif method=='paga_guided_umap':
        embed_center = adata.uns['paga']['pos']
        embed_center_distmat = squareform(pdist(embed_center))

        adjacent_paga = adata.uns['paga']['connectivities'].toarray()
        adjacent_paga_bool = (adjacent_paga >= connect_threshold).astype('int')

        disconnected_cluster_idxs = np.where(np.sum(adjacent_paga_bool, axis=0)==0)[0]
        for dis_idx in disconnected_cluster_idxs:
            cur_argmax = np.argmax(adjacent_paga[:, dis_idx])
            adjacent_paga_bool[dis_idx, cur_argmax] = 1
            adjacent_paga_bool[cur_argmax, dis_idx] = 1

        paga_guided_dist = np.multiply(embed_center_distmat, adjacent_paga_bool)
        G = nx.from_numpy_array(paga_guided_dist)
        ground_distance_mat = -np.inf * np.ones_like(paga_guided_dist)

        len_path = dict(nx.all_pairs_dijkstra_path_length(G))
        for i in len_path.keys():
            for j in len_path[i].keys():
                ground_distance_mat[i,j] = len_path[i][j]
        np.fill_diagonal(ground_distance_mat,0)
        max_val = np.max(ground_distance_mat)
        np.nan_to_num(ground_distance_mat, copy=False, neginf=max_val*2)

    return ground_distance_mat


def build_graph(
    adata,
    coarsen: bool = False,
    coarsen_key: str = None,

    add_landmarks: bool = False,
    landmark_key: str = None,
    landmark_subset: list = None,
    landmark_expression_value: float = 1.0,
    non_landmark_expression_value: float = 0.0,
    landmark_prefix: str = "SpatialLandmark",
    
    add_background: bool = False,
    background_type: str = 'ones',
    background_name: Optional[str] = None,
    background_prefix: str = "SpatialBackground",

    feature_expression_layer: Optional[str] = None,
    feature_aggregate_func: str = 'mean',

    n_neighbors: int = 15,
    use_rep: str = 'X_pca',
    coarsen_method: str = 'paga_guided_umap',
    library_key=None,
    delaunay=False,
    coord_type='generic',
    n_perm: int=10,

    random_state: int = 42,
    **coarsen_kwargs
):
    import squidpy as sq
    import igraph as ig

    adata = adata.copy()

    if not coarsen:

        if use_rep is not 'spatial':
            sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_state)
            connectivities = adata.obsp['connectivities']
        else:
            sq.gr.spatial_neighbors(
                adata, spatial_key=use_rep,
                library_key=library_key,
                coord_type=coord_type,
                n_neighs=n_neighbors,
                delaunay=delaunay
            )
            connectivities = adata.obsp['spatial_connectivities']

        rows, cols = connectivities.nonzero()
        g = ig.Graph(edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
        # if not g.is_connected():
        #     raise RuntimeError(
        #         'The cell-cell graph has disconnected components. Please '
        #         'check your n_neighbors/radius.'
        #     )
        cellular_geodesic_distances = np.array(g.distances())
        if feature_expression_layer is not None:
            adata.X = adata.layers[feature_expression_layer]
    else:
        if coarsen_key is None:
            raise ValueError("When coarsening the graph, coarsen_key must be provided.")

        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, key_added=coarsen_key, random_state=random_state)
        sc.tl.umap(adata, neighbors_key=coarsen_key, random_state=random_state)
        sc.tl.paga(adata, groups=coarsen_key, neighbors_key=coarsen_key)
        
        from scanpy.plotting._tools.scatterplots import (
            _components_to_dimensions, _get_basis
        )
        _basis = _get_basis(adata, basis="X_umap")
        dims = _components_to_dimensions(
            components=None, dimensions=None, total_dims=_basis.shape[1]
        )[0]
        coords = _basis[:, dims]
        pos = (
            pd.DataFrame(coords, columns=["x", "y"], index=adata.obs_names)
            .groupby(adata.obs[coarsen_key], observed=True)
            .median()
            .sort_index()
        ).to_numpy()
        adata.uns["paga"]["pos"] = pos

        cellular_geodesic_distances = get_coarsen_geodesic_distance(
            adata,
            method=coarsen_method,
            connect_threshold=coarsen_kwargs.get("connect_threshold", 0.5)
        )
        adata = sc.get.aggregate(
            adata,
            by=coarsen_key,
            func=feature_aggregate_func,
            layer=feature_expression_layer
        )
        adata.X = adata.layers[feature_aggregate_func]

    if add_landmarks:
        adata = _add_landmarks(
            adata,
            landmark_key=coarsen_key if coarsen else landmark_key,
            landmark_subset=landmark_subset,
            layer=feature_aggregate_func if coarsen else feature_expression_layer,
            landmark_expression_value=landmark_expression_value,
            non_landmark_expression_value=non_landmark_expression_value,
            landmark_prefix=landmark_prefix,
            add_background=add_background,
            background_type=background_type,
            background_name=background_name,
            background_prefix=background_prefix,
            n_perm=n_perm
        )

        
    return cellular_geodesic_distances, adata



def topk_similar(
    query, 
    gdata,
    subset=None,
    ftype=None,
    ftype_key='feature_type', 
    include_query=True,

    query_mode='embedding',
    embedding_key='feature_embedding',
    metric='cosine_similarity',
    dismat_key='feature_distance',

    smaller_is_better=True,
    n=None, metric_threshold=None,
):
    assert query_mode in ['dismat', 'embedding']
    if query not in gdata.obs_names:
        raise ValueError(f"Query item '{query}' not found in gdata.obs_names.")
    if subset:
        gdata = gdata[subset+[query]].copy()
    valid_mask = np.ones(gdata.shape[0], dtype=bool)
    if ftype is not None:
        if isinstance(ftype, str):
            ftype = [ftype]
        # Update the mask to only include items of the specified feature type(s)
        valid_mask &= gdata.obs[ftype_key].isin(ftype).values

    filtered_obs_names = gdata.obs_names[valid_mask]
    query_index_original = gdata.obs_names.get_loc(query)

    if query_mode == 'embedding':
        if embedding_key not in gdata.obsm:
            raise ValueError(f"'{embedding_key}' not found in gdata.obsm.")

        # Get the query embedding and the embeddings of the filtered items
        embeddings = gdata.obsm[embedding_key]
        query_embedding = embeddings[query_index_original].reshape(1, -1)
        filtered_embeddings = embeddings[valid_mask]

        # Calculate distances based on the specified metric
        if metric == 'cosine_distances':
            distances = cosine_distances(query_embedding, filtered_embeddings).flatten()
        elif metric == 'cosine_similarity':
            distances = cosine_similarity(query_embedding, filtered_embeddings).flatten()
        elif metric == 'euclidean':
            distances = euclidean_distances(query_embedding, filtered_embeddings).flatten()
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")
    else: # query_mode == 'dismat'
        if dismat_key not in gdata.uns:
            raise ValueError(f"'{dismat_key}' not found in gdata.uns.")
        # Extract distances from the pre-computed matrix
        distances = gdata.uns[dismat_key][query_index_original, valid_mask].flatten()

    results_df = pd.DataFrame({'metric': distances}, index=filtered_obs_names)

    # Remove the query item from results if include_query is False
    if not include_query and query in results_df.index:
        results_df = results_df.drop(query)

    # Apply the metric_threshold filter first
    if metric_threshold is not None:
        if smaller_is_better:
            results_df = results_df[results_df['metric'] <= metric_threshold]
        else:
            results_df = results_df[results_df['metric'] >= metric_threshold]

    # Sort the DataFrame by distance
    results_df = results_df.sort_values(by='metric', ascending=smaller_is_better)

    # Take the top 'n' results after sorting
    if n is not None:
        results_df = results_df.head(n)

    # Add a rank column based on the final sorted order
    # The rank is simply the order in the sorted DataFrame (1, 2, 3, ...)
    results_df['rank'] = np.arange(1, len(results_df) + 1)

    return results_df



def query_from_multidatasets(
    gdata_list,
    query=None,
    ftype=None,
    include_query=True,
    ftype_key='feature_type',
    query_mode='embedding',
    embedding_key='feature_embedding',
    dismat_key='feature_distance',
    metric='cosine_distances',
    smaller_is_better=True,
    n=None,
    metric_threshold=None
):
    """
    Performs a similarity search across multiple AnnData objects and aggregates results.

    This function first identifies all unique entities across all datasets. It then
    runs a search for each query in each dataset, padding with zero values for any
    entity not found in a given search. The final results are filtered and sorted
    based on the mean metric across all datasets.

    Args:
        gdata_list (list[anndata.AnnData]): A list of AnnData objects to search through.
        query (str or list[str]): A single query item name or a list of names.
        ... (other parameters from topk_similar, except n and metric_threshold are applied post-aggregation)
        n (int, optional): The number of top results to return *after* aggregation and sorting.
        metric_threshold (float, optional): A threshold to filter results based on the mean metric.

    Returns:
        pandas.DataFrame: A DataFrame where the index contains items found to be similar.
        The columns are a MultiIndex with the query item and the metric
        ('count', 'mean_metric', 'mean_rank', 'metrics', 'ranks').
    """
    from collections import defaultdict
    import anndata as ad

    # --- 0. Get all unique entities across all datasets ---
    # If ftype is specified, only collect entities of that type.
    if ftype is None:
        all_entities = pd.Index(
            np.unique(np.concatenate([gdata.obs_names for gdata in gdata_list]))
        )
    else:
        if isinstance(ftype, str):
            ftype = [ftype] # Ensure ftype is a list
        
        filtered_entity_lists = []
        for gdata in gdata_list:
            # Filter obs for the specified feature type(s) and get the names
            matching_entities = gdata.obs_names[gdata.obs[ftype_key].isin(ftype)]
            filtered_entity_lists.append(matching_entities)
        
        all_entities = pd.Index(
            np.unique(np.concatenate(filtered_entity_lists))
        )
    
    # This nested dict will store: {query: {item: {'metrics': [], 'ranks': []}}}
    query_stats = defaultdict(lambda: defaultdict(lambda: {'metrics': [], 'ranks': []}))

    # Ensure query is a list for consistent iteration
    if not isinstance(query, list):
        query = [query]

    # --- 1. Aggregation Step ---
    for q in query:
        for gdata in gdata_list:
            # Check if the query exists in the current dataset
            if q in gdata.obs_names:
                # Run similarity search to get results for this dataset
                query_res_df = topk_similar(
                    query=q,
                    gdata=gdata,
                    ftype=ftype,
                    ftype_key=ftype_key,
                    include_query=include_query,
                    query_mode=query_mode,
                    embedding_key=embedding_key,
                    dismat_key=dismat_key,
                    metric=metric,
                    smaller_is_better=smaller_is_better,
                    n=None, 
                    metric_threshold=None
                )
                
                # For all entities, either record their metric/rank or pad with zero
                for entity in all_entities:
                    if entity in query_res_df.index:
                        row = query_res_df.loc[entity]
                        query_stats[q][entity]['metrics'].append(row['metric'])
                        query_stats[q][entity]['ranks'].append(row['rank'])
                    else:
                        # Entity exists in the combined list, but not in this search result
                        query_stats[q][entity]['metrics'].append(0)
                        query_stats[q][entity]['ranks'].append(0)
            else:
                # If the query itself is not in the dataset, all entities get a zero for this dataset
                for entity in all_entities:
                    query_stats[q][entity]['metrics'].append(0)
                    query_stats[q][entity]['ranks'].append(0)


    # --- 2. Processing and Filtering Step ---
    all_query_dfs = []
    for q, items in query_stats.items():
        if not items:
            continue
            
        # Create a temporary DataFrame for the current query
        temp_df = pd.DataFrame.from_dict(items, orient='index')
        temp_df['count'] = temp_df['metrics'].apply(lambda x: np.count_nonzero(x))
        
        # Calculate mean metric and rank only on non-zero values (i.e., true results)
        temp_df['mean_metric'] = temp_df.apply(lambda row: np.sum(row['metrics']) / row['count'] if row['count'] > 0 else 0, axis=1)
        temp_df['mean_rank'] = temp_df.apply(lambda row: np.sum(row['ranks']) / row['count'] if row['count'] > 0 else 0, axis=1)

        # Apply metric_threshold filter on the mean metric value
        if metric_threshold is not None:
            if smaller_is_better:
                temp_df = temp_df[temp_df['mean_metric'] <= metric_threshold]
            else:
                temp_df = temp_df[temp_df['mean_metric'] >= metric_threshold]
        
        # Sort by the mean metric
        temp_df = temp_df.sort_values(by='mean_metric', ascending=smaller_is_better)
        
        # Take the top 'n' results after sorting
        if n is not None:
            temp_df = temp_df.head(n)

        # Add multi-level columns for combining later
        temp_df.columns = pd.MultiIndex.from_product([[q], temp_df.columns])
        all_query_dfs.append(temp_df)


    if not all_query_dfs:
        return pd.DataFrame()

    # Concatenate all query DataFrames
    final_df = pd.concat(all_query_dfs, axis=1)

    # Reorder columns to be more intuitive
    column_order = ['count', 'mean_metric', 'mean_rank', 'metrics', 'ranks']
    all_cols = []
    for q_col in final_df.columns.get_level_values(0).unique():
        for metric_col in column_order:
            if (q_col, metric_col) in final_df.columns:
                 all_cols.append((q_col, metric_col))
    final_df = final_df[all_cols]
    
    # Fill NaNs in 'count' columns with 0 and convert to integer type
    count_cols = [col for col in final_df.columns if col[1] == 'count']
    final_df[count_cols] = final_df[count_cols].fillna(0).astype(int)

    return final_df
