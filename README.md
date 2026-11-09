# GeneSPOT: Gene SPatial Optimal Transport
This is the *official* implementation of "GeneSPOT: Gene SPatial Optimal Transport".

## Overview
GeneSPOT is a gene-centric computational framework for spatial transcriptomics
analysis that models the spatial deployment of genes across tissue landscapes into
distributed embedding spaces via optimal transport and graph representation
learning. GeneSPOT represents a paradigm shift from traditional cell-centric
approaches to a robust gene-centric framework: it repositions genes as active
participants whose collective spatial behaviors define tissue states and drive
biological processes.

## Preparation
### Installation
```bash
pip install -r requirements.txt
```
> **Note on GPU wheels:** `torch`, `torch-geometric`, and `torch-sparse` are
> CUDA-sensitive. Install the wheels matching your CUDA build *before* running
> `pip install -r requirements.txt` to avoid pip pulling the CPU build or mismatched
> versions. See the
> [PyTorch](https://pytorch.org/get-started/locally/) and
> [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> install guides.


This pulls `palettable`, `statannotations`, `adjustText`, `matplotlib-venn`, `pingouin`,
`gseapy`, and the `NaiveDE` / `SpatialDE` SV-gene methods used in the `reproduce/fig2`
benchmark. R is optional but required by the SEDR / SpatialGlue / GraphST clustering
methods via `rpy2`.

### Prepare Datasets
The notebooks reference the following spatial transcriptomics datasets. Processed
objects are available from the public data repository referenced in the paper's
*Data Availability* statement; raw datasets can be obtained from their original sources:

(1) Mouse prefrontal cortex (mPFC) spatial transcriptomics dataset, from
    ['Three-dimensional intact-tissue sequencing of single-cell transcriptional states'](https://doi.org/10.1126/science.aat5691).
    Used for Figure 2. See `data/mPFC/Dataset11.h5ad`.

(2) Spatial Multimodal Analysis (SMA) of transcriptomes and metabolomes in tissues,
    from ['Spatial multimodal analysis of transcriptomes and metabolomes in tissues'](https://www.nature.com/articles/s41587-023-01937-y).
    Used for Figure 3 (paired and unpaired multi-omics integration). See `data/SMA/`.

(3) Stereo-seq mouse embryo spatiotemporal transcriptomic atlas (MOSTA), from
    ['Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays'](https://doi.org/10.1016/j.cell.2022.04.003).
    Used for Figure 4 (cross-condition spatial dynamics). See `data/stereo_seq_embryo/`.

(4) Spatial aging atlas of the mouse brain, from
    ['Spatial transcriptomic clocks reveal cell proximity effects in brain ageing'](https://www.nature.com/articles/s41586-024-08334-8).
    Used for Figure 5 (aging-atlas comprehensive application). See `data/spatial_aging_clocks/`.

(5) MERFISH mouse brain slices, from
    ['Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region'](https://doi.org/10.1126/science.aau5324).
    Used for the spatially variable gene benchmark in Figure 2. See `data/merfish_5/`.

(6) Xenium human breast cancer dataset, from
    ['High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis'](https://doi.org/10.1038/s41467-023-43458-x).
    Used for the supplementary figures. See `reproduce/supp/xenium_hbc.ipynb`.

The detailed correspondence between subfigures in the manuscript and the Jupyter
notebooks is shown below.

## Reproducing the figures
For each figure, we provide Jupyter notebooks for the main analysis and plotting.
**Before running, please replace the anonymized path placeholders** with your local paths.

| Figure            | Notebook(s)                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------|
| Fig 2             | `reproduce/fig2/fig2.ipynb`                                                                       |
| Fig 3 (paired)    | `reproduce/fig3/paired/1_training_GeneSPOT.ipynb` → `2_fig3cd.ipynb` → `3_fig3e.ipynb` → `4_fig3f.ipynb` |
| Fig 3 (unpaired)  | `reproduce/fig3/unpaired/fig3_unpaired.ipynb`                                                      |
| Fig 4             | `reproduce/fig4/fig4.ipynb`                                                                       |
| Fig 5             | `reproduce/fig5/fig5.ipynb`                                                                       |
| Supplementary     | `reproduce/supp/genespot.ipynb`, `reproduce/supp/xenium_hbc.ipynb`                                |

The paired Fig 3 notebooks are sequential and share a common `res/paired/` directory.
Inline figure outputs are retained inside the notebooks for reference.

## Contact
If you have any questions, please don't hesitate to contact us.
E-mail: [zhiyuan@fudan.edu.cn](mailto:zhiyuan@fudan.edu.cn);
[qizou@mail.sdu.edu.cn](mailto:qizou@mail.sdu.edu.cn).

## Citation

If you find GeneSPOT useful in your research, please cite our paper:

**Zou, Q. et al.** Deciphering the gene-centric spatial dynamics across tissue and time with GeneSPOT. *Nature Methods* (2026). https://doi.org/10.1038/s41592-026-03308-2

```bibtex
@article{zou2026genespot,
  title   = {Deciphering the gene-centric spatial dynamics across tissue and time with GeneSPOT},
  author  = {Zou, Qi and others},
  journal = {Nature Methods},
  year    = {2026},
  doi     = {10.1038/s41592-026-03308-2},
  url     = {https://doi.org/10.1038/s41592-026-03308-2}
}
```
