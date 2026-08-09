#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
FilePath        : /GeneSPOT_rep/genespot/visualization.py
Author          : Qi Zou
Email           : qizou@mail.sdu.edu.cn
Date            : 2025-08-05 04:55:34
-------------------------------------------------
Change Activity :
  LastEditTime  : 2025-08-05 04:55:34
  LastEditors   : Qi Zou & qizou@mail.sdu.edu.cn
-------------------------------------------------
Description     : 
-------------------------------------------------
"""


import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


def create_cmap(color_list, percentage_list=None):
    from matplotlib.colors import LinearSegmentedColormap
    if percentage_list is None:
        percentage_list = np.linspace(0, 1, len(color_list))  # Default transition points

    # Create and return the custom colormap
    cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(percentage_list, color_list)))
    return cmap


def plot_clusters_with_density_and_edges(
    gdata, 
    cluster_key='leiden', basis='X_umap', 
    cluster_subset=None,
    delta=1.0,
    ax=None,
    save=False,
    save_path='.',
    save_name='umap_with_kde',
    legend=True,
    show=False,
    scatter_edgecolor='white',
    contour_linewidths=2, scatter_linewidths=0.5, scatter_nan_linewidths=0.2,
    scatter_s=20, scatter_nan_s=5, na_color='none', connectivity_color='grey',
    scatter_alpha=0.7,
    draw_edges=True
):
    if ax is None:
        fig, ax = plt.subplots()
    gclusters = gdata.obs[cluster_key].cat.categories
    
    if cluster_subset is None:
        cluster_subset = gclusters
    elif not isinstance(cluster_subset, list):
        cluster_subset = [cluster_subset]
        
    # Plot density contours and scatter points for each cluster
    for i, gc in enumerate(gclusters):
        plot_cluster_density(
            ax, 
            gdata=gdata, 
            cluster_key=cluster_key,
            basis=basis,
            cluster=gc,
            color=gdata.uns[f"{cluster_key}_colors"][int(gc)] if gc in cluster_subset else na_color,
            delta=delta,
            contour_linewidths=contour_linewidths,
            scatter_linewidths=scatter_linewidths if gc in cluster_subset else scatter_nan_linewidths,
            show_density=gc in cluster_subset,
            s=scatter_s if gc in cluster_subset else scatter_nan_s,
            edgecolors=scatter_edgecolor, alpha=scatter_alpha
        )
    
    # Draw connectivity edges
    if draw_edges:
        draw_connectivity_edges(ax, gdata=gdata, basis=basis, c=connectivity_color)
    # Add legend
    if legend:
        add_cluster_legend(ax, gdata=gdata, cluster_key=cluster_key)
    ax.set_axis_off()
    
    if save:
        plt.savefig(f'{save_path}/{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()

def plot_cluster_density(
    ax, gdata, cluster_key,
    basis, cluster, color, delta,
    s=15, contour_linewidths=2, 
    scatter_linewidths=0.5,
    edgecolors='black',
    show_density=True, alpha=0.9
):
    genes = gdata.obs_names[gdata.obs[cluster_key] == cluster]
    x = gdata[genes].obsm[basis][:, 0]
    y = gdata[genes].obsm[basis][:, 1]

    if show_density:
        # Density grid limits
        xmin, xmax = x.min() - delta, x.max() + delta
        ymin, ymax = y.min() - delta, y.max() + delta

        # Compute density
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        levels = np.linspace(f.min(), f.max(), 10)

        # Plot density contour
        ax.contour(
            xx, yy, f,
            levels=levels,
            cmap=create_cmap(['#ffffff', color, color], [0, 0.75, 1]),
            # cmap=create_cmap([color, color], [0, 1]),
            linewidths=contour_linewidths,
        )

    # Plot scatter points
    ax.scatter(x, y, 
               s=s,
               linewidths=scatter_linewidths,
               color=color, 
               edgecolors=edgecolors,
               alpha=alpha,
               zorder=80)

def draw_connectivity_edges(ax, gdata, basis, c='grey'):
    """
    Draw edges between connected points based on the adjacency matrix.
    """
    connectivities = gdata.obsp["connectivities"].toarray()
    umap_coords = gdata.obsm[basis]
    
    for i in range(connectivities.shape[0]):
        for j in range(i + 1, connectivities.shape[1]):  # Traverse upper triangle of the adjacency matrix
            if connectivities[i, j] > 0:  # If nodes are connected
                ax.plot([umap_coords[i, 0], umap_coords[j, 0]], 
                        [umap_coords[i, 1], umap_coords[j, 1]],
                        c=c, linewidth=connectivities[i, j] * 2,
                        zorder=1, alpha=0.5)

def add_cluster_legend(ax, gdata, cluster_key):
    """
    Add legend to the plot for clusters.
    """
    cluster_legends = []
    for i, cluster in enumerate(range(gdata.obs[cluster_key].unique().shape[0])):
        cluster_legends.append(
            Line2D([0], [0], marker='o', color=gdata.uns[f'{cluster_key}_colors'][i], 
                   markerfacecolor=None, linestyle='None', markersize=8,
                   label=cluster)
        )
    ax.legend(handles=cluster_legends, title='Gene Cluster', title_fontsize=12, 
              bbox_to_anchor=(1, -1), loc="lower center", frameon=False, ncols=4)
    
    
def legend_handles(items, colors, marker='o', markeredgecolor=None, markersize=8, linestyle='None'):
    legends = []
    for i, item in enumerate(items):
        legends.append(
            Line2D(
                [], [], 
                color=colors[i],
                marker=marker, markeredgecolor=markeredgecolor, markersize=markersize,
                linestyle=linestyle,
                label=item,
            )
        )
    return legends


def color_by_scores(scores, cmap='Reds'):
    import matplotlib.colors as mcolors
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    return [mcolors.to_hex(cmap(norm(d))) for d in scores]


def soview(
    adata, 
    layer=None,
    embedding_key=None,
    r=None, g=None, b=None,
    size_factor=0,
    s=1,
    spatial_key='spatial',
    marker='o',
    ax=None,
    white_ratio=0.0,
    facecolor='black',
    frameon=False
):
    # Check that either embedding_use or r,g,b are provided
    if embedding_key is None and all(x is None for x in [r, g, b]):
        raise ValueError("Either embedding_use or at least one of r, g, b must be provided")

    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if embedding_key is not None:
        rgb = adata.obsm[embedding_key]
        channel_names = ['R', 'G', 'B']
    else:
        # Create RGB array, filling None channels with white (1.0)
        rgb_channels = []
        channel_names = []
        for i, (channel, name) in enumerate(zip([r, g, b], ['R', 'G', 'B'])):
            if channel is None:
                # White channel (will be inverted to black later)
                rgb_channels.append(np.ones((adata.n_obs)))
                channel_names.append(f'{name} (empty)')
            else:
                # val = adata.obs[channel].values if channel in adata.obs
                if channel in adata.obs:
                    val = adata.obs[channel].values
                elif channel in adata.var_names:
                    val = adata[:, channel].X if layer is None else adata[:, channel].layers[layer]
                else:
                    raise ValueError(f"Channel '{channel}' not found in adata.obs or adata.var_names")
                    
                if issparse(val):
                    val = val.toarray()
                rgb_channels.append(val.reshape(-1, ))
                channel_names.append(f'{name}: {channel}')

        rgb = np.stack(rgb_channels, axis=1)

    # Normalize to 0-1 range
    rgb = MinMaxScaler(clip=True).fit_transform(rgb)

    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6), facecolor=facecolor)
        ax.set_facecolor(facecolor)
    else:
        fig = ax.figure
        fig.patch.set_facecolor(facecolor)
        ax.set_facecolor(facecolor)
        
    ax.scatter(
        x=adata.obsm[spatial_key][:,0],
        y=adata.obsm[spatial_key][:,1], 
        c=rgb,
        edgecolors='none',
        marker=marker,
        s=rgb.max(axis=1) * size_factor if size_factor is not None else s
    )
    
    if not frameon:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return ax