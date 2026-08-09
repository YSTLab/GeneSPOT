#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /GeneSPOT_rep/genespot/utils.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2025-08-04 11:21:27
-------------------------------------------------
Change Activity :
  LastEditTime  : 2025-08-04 11:22:42
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse


def seed_everything(seed=42):
    """
    Seed all random number generators to ensure reproducibility.

    Parameters:
    seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def merge_bin_coordinate(coordinate: np.ndarray, coordinate_min: float, bin_size: float) -> np.ndarray:
    """
    将连续坐标转换为离散的bin坐标。
    
    Parameters
    ----------
    coordinate : np.ndarray
        原始坐标数组
    coordinate_min : float
        坐标的最小值（起始点）
    bin_size : float
        每个bin的大小
        
    Returns
    -------
    np.ndarray
        离散化后的bin坐标（整数数组）
    """
    return np.floor((coordinate - coordinate_min) / bin_size).astype(int)


def get_bin_center(bin_coordinate: np.ndarray, coordinate_min: float, bin_size: float) -> np.ndarray:
    """
    从bin坐标计算bin中心的实际坐标。
    
    Parameters
    ----------
    bin_coordinate : np.ndarray
        离散的bin坐标
    coordinate_min : float
        坐标的最小值（起始点）
    bin_size : float
        每个bin的大小
        
    Returns
    -------
    np.ndarray
        bin中心的实际坐标
    """
    return bin_coordinate * bin_size + coordinate_min + bin_size / 2


def bin_spatial_adata(adata: ad.AnnData, bin_size: float = 50, method: str = "sum") -> ad.AnnData:
    """
    将空间转录组数据按空间位置分箱聚合。
    
    Parameters
    ----------
    adata : ad.AnnData
        输入的AnnData对象，必须包含 'spatial' 在 obsm 中
    bin_size : float, default=50
        分箱的大小（空间单位）
    method : str, default="sum"
        聚合方法，支持: 'sum', 'mean', 'median', 'max', 'min'
        
    Returns
    -------
    ad.AnnData
        分箱后的新AnnData对象，每个观测对应一个空间bin
        
    Raises
    ------
    ValueError
        如果 method 不在支持的方法列表中
        如果 adata 中不包含 'spatial' 坐标
    """
    # 参数验证
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm must contain 'spatial' coordinates")
    
    valid_methods = ["sum", "mean", "median", "max", "min"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    
    # 计算网格坐标
    coords = adata.obsm["spatial"]
    grid_x = (coords[:, 0] // bin_size).astype(int)
    grid_y = (coords[:, 1] // bin_size).astype(int)
    
    # 分组聚合
    grid_df = pd.DataFrame({
        "grid_x": grid_x,
        "grid_y": grid_y,
        "index": np.arange(len(adata))
    })
    grouped = grid_df.groupby(["grid_x", "grid_y"], sort=True)
    
    # 预处理：统一处理稀疏矩阵
    X = adata.X
    is_sparse = issparse(X)
    
    # 预分配数组
    n_bins = len(grouped)
    n_features = adata.shape[1]
    new_X = np.zeros((n_bins, n_features))
    new_coords = np.zeros((n_bins, 2))
    new_obs_names = []
    
    # 选择聚合函数
    agg_func_map = {
        "sum": np.sum,
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min
    }
    agg_func = agg_func_map[method]
    
    # 遍历每个bin进行聚合
    for i, ((gx, gy), group) in enumerate(grouped):
        indices = group["index"].values
        sub_X = X[indices]
        
        # 转换稀疏矩阵为密集矩阵
        if is_sparse:
            sub_X = sub_X.toarray()
        
        # 聚合
        new_X[i] = agg_func(sub_X, axis=0).ravel()
        
        # 计算bin中心坐标
        new_coords[i] = [(gx + 0.5) * bin_size, (gy + 0.5) * bin_size]
        new_obs_names.append(f"bin_{gx}_{gy}")
    
    # 创建新的AnnData对象
    new_adata = ad.AnnData(
        X=new_X,
        var=adata.var.copy(),
        uns=adata.uns.copy()
    )
    new_adata.obsm["spatial"] = new_coords.astype(int)
    new_adata.obs_names = new_obs_names
    new_adata.obs['x'] = new_coords[:, 0].astype(int)
    new_adata.obs['y'] = new_coords[:, 1].astype(int)
    
    return new_adata



def filter_extreme_values(adata, layer=None, lower_quantile=0.05, upper_quantile=0.95):
    if layer is None:
        if isinstance(adata, sc.AnnData):
            expression_matrix = adata.X.copy()
        elif isinstance(adata, np.ndarray):
            expression_matrix = adata.copy()
    else:
        expression_matrix = adata.layers[layer].copy()

    if issparse(expression_matrix):
        expression_matrix = expression_matrix.toarray()

    # 对每个基因过滤掉极端值
    for i in range(expression_matrix.shape[1]):
        gene_expression = expression_matrix[:, i].flatten()
        lower_bound = np.percentile(gene_expression, lower_quantile * 100)
        upper_bound = np.percentile(gene_expression, upper_quantile * 100)
        expression_matrix[:, i] = np.clip(gene_expression, lower_bound, upper_bound)

    return expression_matrix