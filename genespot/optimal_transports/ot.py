#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /GeneSPOT_rep/src/optimal_transports/ot.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2025-08-04 11:27:25
-------------------------------------------------
Change Activity :
  LastEditTime  : 2025-08-04 11:28:16
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : Compute the Earth Mover's Distance (EMD) matrix between gene expression vectors.
                  Reference implementation: https://github.com/KlugerLab/GeneTrajectory-python/blob/main/gene_trajectory/gene_distance_shared.py
-------------------------------------------------
"""


from typing import Optional, Sized
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import SharedMemoryManager

import ot
import numpy as np
from tqdm import tqdm

from .shared_array import PartialStarApply, SharedArray


_DEFAULT_NUMITERMAX = 50000


def cal_ot_distance(
    gene_expr, ot_cost,
    gene_pairs: Optional[Sized] = None,
    num_iter_max=_DEFAULT_NUMITERMAX,
    show_progress_bar=True,
    processes: Optional[int] = None
):
    assert gene_expr.shape[0] == ot_cost.shape[0]
    ncells, ngenes = gene_expr.shape
    
    if gene_pairs is None:
        pairs = ((i, j) for i in range(0, ngenes - 1) for j in range(i + 1, ngenes))
        npairs = (ngenes * (ngenes - 1)) // 2
    else:
        pairs = gene_pairs
        npairs = len(gene_pairs)
    
    emd_mat = np.full((ngenes, ngenes), fill_value=np.nan)

    if processes is not None and processes > 1:
        with SharedMemoryManager() as manager:
            with ThreadPoolExecutor(max_workers=processes) as pool:
                # prepare shared environment
                cost = SharedArray.copy(manager, np.asarray(ot_cost))
                gexp = SharedArray.copy(manager, np.asarray(gene_expr))
                f = PartialStarApply(_cal_ot, cost, gexp)

                # execute tasks and process results
                result_generator = pool.map(f,  ((i, j, num_iter_max) for i, j in pairs))
                if show_progress_bar:
                    result_generator = tqdm(result_generator, total=npairs, position=0, leave=True)
                for d, i, j in result_generator:
                    emd_mat[i, j] = emd_mat[j, i] = d
    else:    
        for i, j in tqdm(pairs, disable=not show_progress_bar, total=npairs):
            dis, i, j = _cal_ot(
                ot_cost_matrix=ot_cost,
                gene_expr_matrix=gene_expr,
                i=i,
                j=j,
                num_iter_max=num_iter_max,
            )
            emd_mat[i][j] = emd_mat[j][i] = dis
        
    np.fill_diagonal(emd_mat, 0)
    np.nan_to_num(emd_mat, nan=1000 * np.nanmax(emd_mat), copy=False)
            
    return emd_mat


def _cal_ot(ot_cost_matrix: np.array, gene_expr_matrix: np.array, i: int, j: int, num_iter_max: int = _DEFAULT_NUMITERMAX):
    """Computes the Earth Mover's Distance (EMD) between two gene expression vectors."""
    gene_i = np.array(gene_expr_matrix[:, i]).reshape(-1)
    gene_j = np.array(gene_expr_matrix[:, j]).reshape(-1)

    gene_i = gene_i / sum(gene_i)
    gene_j = gene_j / sum(gene_j)

    emd_dist = ot.emd2(gene_i[np.nonzero(gene_i)],
                       gene_j[np.nonzero(gene_j)],
                       ot_cost_matrix[np.nonzero(gene_i)[0], :][:, np.nonzero(gene_j)[0]],
                       numItermax=num_iter_max)
    
    return emd_dist, i, j
