#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /GeneSPOT/genespot/graph_representation/node2vec.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2025-08-04 11:27:26
-------------------------------------------------
Change Activity :
  LastEditTime  : 2025-08-04 11:27:44
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import networkx as nx
import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch
from tqdm import tqdm


def node2vec(
    cod,
    n_neighbors=5,
    dimensions=30,
    walk_length=5,
    context_size=5,
    num_walks=200,
    workers=5,
    batch_size=128,
    epochs=200,
    p=1.0, q=0.5,
    num_negative_samples=1,
    lr=1e-3,
    device='cpu',
    norm=True,
    show_progress_bar=True,
):
    from torch_geometric.nn import Node2Vec

    num_nodes = cod.shape[0]
    # Create k-nearest neighbors graph using sklearn
    knn_graph =  kneighbors_graph(cod, n_neighbors=n_neighbors, include_self=False, metric="precomputed", mode='distance')
    # Extract edge indices from the sparse matrix
    source, target = np.array(knn_graph.nonzero())
    edge_index = torch.from_numpy(
        np.stack([
            source,
            target], axis=0),
        ).long().contiguous()

    device = torch.device('cpu' if not torch.cuda.is_available() else device)
    model = Node2Vec(
        edge_index=edge_index,
        num_nodes=num_nodes,
        embedding_dim=dimensions,
        walk_length=walk_length,
        context_size=context_size,  # Typically set to the same as walk_length
        walks_per_node=num_walks,
        num_negative_samples=num_negative_samples,
        p=p,  # Return hyperparameter
        q=q,  # In-out hyperparameter
        sparse=True
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    # Training loop
    model.train()
    loss_list = []
    
    with tqdm(range(epochs), desc='Training', disable=not show_progress_bar) as pbar:
        for epoch in pbar:
            total_loss = 0
            # pbar.set_description(f'EPOCH: {epoch}')
            
            for pos_rw, neg_rw in model.loader(batch_size=batch_size, shuffle=True, num_workers=workers):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            loss_list.append(total_loss)
            pbar.set_postfix(loss=total_loss)
        

    # Get the embeddings
    model.eval()
    embeddings = model().detach().cpu().numpy()

    if norm:
        norms = np.linalg.norm(embeddings, axis=1)
        embeddings = embeddings / norms[..., np.newaxis]

    return embeddings, model, loss_list