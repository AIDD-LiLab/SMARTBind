# SMARTBind

This is the official repository for SMARTBind model. SMARTBind is a sequence-based RNA-ligand interaction prediction 
method by leveraging contrastive learning and RNA foundation model. It can be used for RNA-ligand interaction virtual screening
and binding site prediction for the potential ligand binders.

## Installation
```bash
git clone https://github.com/AIDD-LiLab/SMARTBind.git
cd SMARTBind
```

### Install Required Packages
To install the required packages, run the following command:
```bash
# cuda 12.4 is required to satisfy the torch version
conda env create -f environment.yml
conda activate SMARTBind_env
```

## Inference with SMARTBind
Downloaded pretrained SMARTBind models from [Zenodo](https://zenodo.org/records/17197893). 
An alternative way is to use `gdown` as follows if you are working in a server without browser access:
```bash
mkdir SMARTBind_weight
cd SMARTBind_weight
gdown --id 1z0PD0CRMAs1Q43g836JMzh0VFcAcoG-l
unzip SMARTBind_weight.zip
rm SMARTBind_weight.zip
```

Please refer to the `notebook/README.md` for the details of the inference with SMARTBind model using jupyter
notebooks.

## Acknowledgements
We thank the authors of following open-source packages and models: [RNA-FM](https://github.com/ml4bio/RNA-FM), 
[BioPython](https://biopython.org/), [ProDy](http://prody.csb.pitt.edu/), 
[Open Babel](https://openbabel.org/index.html), [RDKit](https://www.rdkit.org/), [RNA 3D Hub](http://rna.bgsu.edu/rna3dhub/),
[DeepCoy](https://github.com/fimrie/DeepCoy), [RNA3DB](https://github.com/marcellszi/rna3db), [MMseqs2](https://github.com/soedinglab/MMseqs2),
[PyTorch Lightning](https://www.pytorchlightning.ai/).


## Cite SMARTBind
```bibtex
@article{,
title={},
author={},
journal={bioRxiv},
year={2025},
publisher={Cold Spring Harbor Laboratory}
}
```
