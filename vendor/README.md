# vendor

This directory vendors third-party / external niche-embedding methods used only for
benchmarking and comparison in the reproduction notebooks. These packages are **not**
part of GeneSPOT itself and are **not tracked by git** (see `.gitignore` at the repo
root). They are kept locally so the reproduction scripts can import them directly.

## Contents

`vendor/niche_embedding/` — spatial niche-embedding baselines:

- `BINARY`
- `CAST`
- `GraphST`
- `MENDER`
- `SEDR`
- `STAGATE`
- `SpatialGlue`
- `cellcharter`
- `scconfluence`

## Notes

- These subpackages retain their original upstream code (with only minimal local
  patches where needed to run in this environment). Please refer to each method's
  upstream repository for licensing and citation information.
- Because this directory is git-ignored, cloning the repository will NOT fetch these
  baselines. If you need to reproduce the benchmark comparisons, obtain the
  corresponding packages from their original sources and place them under
  `vendor/niche_embedding/` following the same layout.
