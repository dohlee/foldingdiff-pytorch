# foldingdiff-pytorch

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning)

![banner](img/banner.png)

An unofficial re-implementation of FoldingDiff, a diffusion-based generative model for protein backbone structure generation.

## Installation

WIP

## Downloading and preprocessing training data
Download non-redundant protein backbone structure data (40% similary cutoff) from CATH.
```bash 
$ wget ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz
```

Extract the downloaded file and attach `.pdb` extension to files
```bash
$ tar xvf cath-dataset-nonredundant-S40.pdb.tgz && cd dompdb
$ for f in *; do mv "$f" "$f.pdb"; done
```

Run `snakemake` pipeline to convert pdb files to `npy` files containing angle information of shape (n, 6).
```
$ snakemake -s preprocess.smk -prq -j [CORES] --keep-going
```