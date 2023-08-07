# Proximal Exploration (PEX)

This repository contains a PyTorch implementation of our paper [Proximal Exploration for Model-guided Protein Sequence Design](https://www.biorxiv.org/content/10.1101/2022.04.12.487986) published at ICML 2022.
Proximal Exploration (PEX) is a variant of directed evolution, which prioritizes the search for low-order mutants.
Based this local-search mechanism, a model architecture called Mutation Factorization Network (MuFacNet) is developed to specialize in the local fitness landscape around the wild type.

## Installation

The dependencies can be set up using the following commands:

```bash
conda create -n pex python=3.10 -y
conda activate pex
conda install pytorch pytorch-cuda -c pytorch -c nvidia -y
conda install -c conda-forge tape_proteins=0.5 -y
pip install -e .
```

If you run into dependency version issues, try `pip install -r requirements.txt` before running `pip install -e .`.

Clone this repository and download the oracle landscape models by the following commands:

```bash
git clone https://github.com/HeliXonProtein/proximal-exploration.git
cd proximal-exploration
bash download_landscape.sh
```

## Usage

Run the following commands to reproduce our main results shown in section 5.1. There are eight fitness landscapes to support a diverse evaluation on black-box protein sequence design.

```bash
python run.py --alg=pex --net=mufacnet --task=avGFP  # Green Fluorescent Proteins
python run.py --alg=pex --net=mufacnet --task=AAV    # Adeno-associated Viruses
python run.py --alg=pex --net=mufacnet --task=TEM    # TEM-1 β-Lactamase
python run.py --alg=pex --net=mufacnet --task=E4B    # Ubiquitination Factor Ube4b
python run.py --alg=pex --net=mufacnet --task=AMIE   # Aliphatic Amide Hydrolase
python run.py --alg=pex --net=mufacnet --task=LGK    # Levoglucosan Kinase
python run.py --alg=pex --net=mufacnet --task=Pab1   # Poly(A)-binding Protein
python run.py --alg=pex --net=mufacnet --task=UBE2I  # SUMO E2 conjugase
```

In the default configuration, the protein fitness landscape is simulated by a TAPE-based oracle model. By adding the argument `--oracle_model=esm1b`, the landscape simulator is switched to an oracle model based on ESM-1b.

## Contact

Please contact zhizhour[at]helixon.com for any questions related to the source code.
