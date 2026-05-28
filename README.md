# SMARTBind

This is the official repository for SMARTBind model. SMARTBind is a sequence-based RNA-ligand interaction prediction 
method by leveraging contrastive learning and RNA foundation model. It can be used for RNA-ligand interaction virtual screening
and binding site prediction for the potential ligand binders.


## System requirements

### Software dependencies

SMARTBind is distributed as a conda environment. Exact pinned versions are in
`environment.yml`. Key components:

| Component         | Version                                       |
|-------------------|-----------------------------------------------|
| Operating system  | Linux x86-64 (Ubuntu 20.04 / 22.04 LTS)       |
| Python            | 3.9.21                                        |
| PyTorch           | 2.5.1 (CUDA 12.4 build)                       |
| CUDA runtime      | 12.4                                          |
| cuDNN             | 9.1.0                                         |
| PyTorch Lightning | 2.5.0.post0                                   |
| RDKit             | 2024.09.4                                     |
| BioPython         | 1.84                                          |
| Open Babel        | 3.1.1                                         |
| Hydra             | 1.3.2                                         |

macOS and native Windows are **not** supported by the shipped `environment.yml`
(which is pinned to `linux-64`). Reviewers on macOS or Windows should use WSL2
or a Linux container.

### Tested on

| OS                                        | Python  | PyTorch | CUDA | GPU                          | Driver |
|-------------------------------------------|---------|---------|------|------------------------------|--------|
| Ubuntu 22.04.5 LTS (native)               | 3.9.21  | 2.5.1   | 12.4 | NVIDIA RTX 3090 (24 GB)      | 591.86 |
| Ubuntu 22.04.5 LTS (WSL2 on Windows 11)   | 3.9.21  | 2.5.1   | 12.4 | NVIDIA RTX 3090 (24 GB)      | 591.86 |

(Maintainers: add A100 / H100 / additional tested GPUs as they become available.)

### Hardware requirements

| Resource     | Minimum                                                | Recommended                    | Notes                                                                                       |
|--------------|--------------------------------------------------------|--------------------------------|---------------------------------------------------------------------------------------------|
| GPU          | NVIDIA, ≥ 8 GB VRAM, compute capability ≥ 7.0          | RTX 3090 / A100, 24+ GB VRAM   | Required for training. Inference can fall back to CPU with `--device cpu`.                  |
| System RAM   | 8 GB                                                   | 16 GB                          | Conda environment solve peaks at ≈ 4 GB.                                                    |
| Disk space   | 12 GB free                                             | 25 GB free                     | Conda env (~ 5 GB) + weights (~ 2.6 GB) + Zenodo bundle (~ 2.6 GB) + training outputs.       |
| Network      | HTTPS to github.com, zenodo.org, drive.google.com      | —                              | RNA-FM weights additionally require reachability to `proj.cse.cuhk.edu.hk` unless mirrored. |
| CPU          | 4 cores                                                | 8+ cores                       | Affects only SMILES → FP2 preprocessing throughput.                                          |


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

### Typical install time

On a workstation with a 100 Mbps internet link and an NVMe-class SSD:

| Step                                                          | Wall-clock      |
|---------------------------------------------------------------|-----------------|
| `conda env create -f environment.yml` (libmamba solver)       | ~ 7–10 min      |
| Pretrained-weight download from Zenodo (~ 2.6 GB)             | ~ 5 min         |
| Pretrained-weight download via gdown (~ 2.6 GB, alternative)  | ~ 10–15 min     |
| **Total cold install**                                        | **~ 15–25 min** |

For faster solves, install libmamba first:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
### RNA-FM weights (one-time)

SMARTBind uses RNA-FM as its RNA encoder. On first run, the code attempts to
download RNA-FM weights from `https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth`.
If your installation errors with `HTTP 403` or hangs at the RNA-FM download
step, place the weight file manually:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
# Place RNA-FM_pretrained.pth here (≈ 398 MB).
# Authors: replace this comment with a persistent (Zenodo) mirror once available.
```

## Inference with SMARTBind
Downloaded pretrained SMARTBind models from [Zenodo](https://zenodo.org/records/20252150). 
An alternative way is to use `gdown` as follows if you are working in a server without browser access:
```bash
mkdir SMARTBind_weight
cd SMARTBind_weight
gdown --id 1z0PD0CRMAs1Q43g836JMzh0VFcAcoG-l
unzip SMARTBind_weight.zip
rm SMARTBind_weight.zip
```

### Binding score prediction and virtual screening

SMARTBind takes one RNA sequence and a ligand library in SMILES format, then
ranks the ligands by the ensemble-averaged RNA-ligand binding score. The ligand
library should be a text file with one SMILES string per line. A minimal example
based on `notebook/virtual_screening.ipynb` is:

```python
import pandas as pd
import tqdm
from torch.nn.functional import cosine_similarity

from smartbind import load_smartbind_models
from smartbind.preprocess import convert_smiles_to_pf2

input_rna_chain = "GACAGCUGCUGUC"
smiles_txt_path = "notebook/ligand_library.txt"
ensembled_models_path = "SMARTBind_weight"
save_path = "binding_score.txt"
device = "cpu"
batch_size = 10000

with open(smiles_txt_path, "r") as f:
    smiles_list = f.read().splitlines()

smol_fp2_list = [convert_smiles_to_pf2(smiles) for smiles in smiles_list]
smartbind_models = load_smartbind_models(
    model_path=ensembled_models_path,
    device=device,
    vs_mode=True,
)

rank_result_by_models = {}
for model_idx, model in enumerate(tqdm.tqdm(smartbind_models)):
    rna_embed = model.inference_single_rna(input_rna_chain)
    rank_result_by_models[model_idx] = []

    for start in range(0, len(smol_fp2_list), batch_size):
        end = min(start + batch_size, len(smol_fp2_list))
        ligand_embeds = model.inference_list_smols(smol_fp2_list[start:end])
        scores = cosine_similarity(rna_embed, ligand_embeds).tolist()
        rank_result_by_models[model_idx].extend(scores)

df = pd.DataFrame(rank_result_by_models)
df["average"] = df.mean(axis=1)
df.index = smiles_list
df[["average"]].sort_values("average", ascending=False).to_csv(
    save_path,
    sep="\t",
    header=["Binding_Score"],
    index_label="Ligand_ID",
)
```

### Binding site prediction

SMARTBind also predicts residue-level binding scores for a given RNA-ligand
pair. The following minimal example is adapted from
`notebook/binding_site_prediction.ipynb` and returns one normalized score per
RNA residue:

```python
import os

import numpy as np
from smartbind import BindingPL
from smartbind.preprocess import convert_smiles_to_pf2

def predict_binding_site(target_rna_chain, ligand_smiles, model_dir="SMARTBind_weight", device="cpu"):
    ligand_fp2 = convert_smiles_to_pf2(ligand_smiles)
    binding_site_predictions = {i: [] for i in range(len(target_rna_chain))}

    for weight_name in os.listdir(model_dir):
        if not weight_name.endswith(".pth"):
            continue
        weight_path = os.path.join(model_dir, weight_name)
        model = BindingPL(device=device, vs_mode=True)
        model.load_pretrained_model(model_path=weight_path, device=device, mode="inference")

        pred = model.predict_binding(target_rna_chain, ligand_fp2).detach().cpu().numpy()
        pred = (pred - min(pred)) / (max(pred) - min(pred))
        for i, score in enumerate(pred):
            binding_site_predictions[i].append(score)

    average_predictions = [
        sum(binding_site_predictions[i]) / len(binding_site_predictions[i])
        for i in range(len(target_rna_chain))
    ]
    average_predictions = np.array(average_predictions)
    normalized_predictions = (
        (average_predictions - min(average_predictions)) /
        (max(average_predictions) - min(average_predictions))
    )
    return [float(round(score, 3)) for score in normalized_predictions]

rna = "GACAGCUGCUGUC"
ligand = "c1cc(ccc1C2=NCCN2CCN)C(=O)Nc3ccc(cc3)C4=NCCN4"
binding_site_scores = predict_binding_site(rna, ligand)
print(binding_site_scores)
```

Please refer to the `notebook/README.md` for the details of the inference with SMARTBind model using jupyter
notebooks.

## Training Dataset
### Preprocessed Dataset
The processed dataset used for this work can be downloaded from the [Zenodo](https://zenodo.org/records/20252150).
- 10-fold RNAmigos1 random split training dataset can be downloaded from https://drive.google.com/file/d/1iQBwmtlKUzaxSdl58oEeruMSskoJcXzy/view?usp=sharing.
- 10-fold HARIBOSS random split training datasets can be downloaded from https://drive.google.com/file/d/1oyf7Tr2I_Yx2vHibymb1_fi0Lt8SiK_0/view?usp=sharing.
- All the splits for HARIBOSS training datasets can be downloaded from https://drive.google.com/drive/folders/1j7QMGHKyhpJLasyXLjYZ2Db9SbvKNKbR?usp=sharing.
- All the splits for the GerNA-Bind subset of the HARIBOSS training datasets can be downloaded from https://drive.google.com/drive/folders/1V5jDPI4scKlNuW9RbB7uS90xDNT15Oke?usp=sharing.

To pretraining the model, please download the dataset and put it in the `data` folder.

## Train the SMARTBind from Scratch
For training SMARTBind model, there are two stages of training. The binding score model is trained from 
scratch with fine-tuning the RNA-FM model. The binding site prediction model is trained with the 
pre-trained binding score model.

### Stage 1: train the binding score prediction task with RNA-FM fine-tuned.
1. Modify `binding_score_training.yaml` in `conf` folder.
2. Set `data_params.training_data` to the path to the training data downloaded from Training Data Section.

3. Execute the following command.
```bash
python train_binding_score.py
```
#### Training with ligand-specific decoy augmentation
Please refer to the `notebook/decoy_augmentation.ipynb` for building decoy augmented training data, processed `hariboss_merged_10fd_with_decoys.pkl`
is also provided. By modifying `data_params.decoy_num` and `data_params.extra_decoy_num` to adjust the proportion of native
negative decoys and augmented decoys during training, our default setting is 48:24. By setting 72:0, it will be equivalent to training without decoy augmentation.

### Stage 2: train the binding site prediction task with pre-trained binding score prediction model.
If you want to retrain a SMARTBind binding site prediction model, it's recommended to use the trained binding score model
from Stage 1 or the provided pre-trained binding score model in `SMARTBind_weight` folder to initialize the binding score module weights.

1. Modify `binding_site_training.yaml` in `conf` folder.
2. Set `data_params.training_data` to the path to the training data.
If you want to use your own dataset, please follow the jupyter notebook in `notebook/binding_data_curation.ipynb` 
folder to preprocess your dataset.
3. Set `model_params.binding_score_model_path` to the path to the folder containing the pre-trained binding score model.
4. Execute the following command.

```bash
python train_binding_site.py
```


## Acknowledgements
We thank the authors for making the following packages, software, and models open-sourced and easy to implement:

- RNA foundation model: [RNA-FM](https://github.com/ml4bio/RNA-FM)
- RNA and molecular analysis: [BioPython](https://biopython.org/), [ProDy](http://prody.csb.pitt.edu/), [RNA 3D Hub](http://rna.bgsu.edu/rna3dhub/), [RNA3DB](https://github.com/marcellszi/rna3db), [MMseqs2](https://github.com/soedinglab/MMseqs2), [Open Babel](https://openbabel.org/index.html), [RDKit](https://www.rdkit.org/), [RMalign]()
- Decoy generation: [DeepCoy](https://github.com/fimrie/DeepCoy), [DecoyFinder](https://github.com/URV-cheminformatics/DecoyFinder)
- Binding score and virtual-screening benchmark methods: [RNAmigos1](https://github.com/cgoliver/RNAmigos), [RNAmigos2](https://github.com/cgoliver/rnamigos2), [GerNA-Bind](https://github.com/GENTEL-lab/GerNA-Bind), [RNAsmol](https://github.com/hongli-ma/RNAsmol), [SMRTnet](https://github.com/Yuhan-Fei/SMRTnet), [DrugBAN](https://github.com/peizhenbai/DrugBAN), [GraphDTA](https://github.com/thinng/GraphDTA)
- Binding site benchmark methods: [fpocketR](https://github.com/Weeks-UNC/fpocketR), [RNAsite](https://academic.oup.com/bioinformatics/article/37/1/36/6069564), [RLsite](https://github.com/SaisaiSun/RLsite), [Rsite2](https://pubmed.ncbi.nlm.nih.gov/26751501/)
- Docking and virtual screening: [AutoDock](https://autodocksuite.scripps.edu/autodock4/), [rDock](https://rdock.github.io/), [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU), [GNINA 1.3](https://github.com/gnina/gnina), [DeepDocking](https://github.com/jamesgleave/DD_protocol)
- Deep learning framework: [PyTorch Lightning](https://www.pytorchlightning.ai/)


## Cite SMARTBind
```bibtex
@article {Jiang2025.09.24.678312,
	author = {Jiang, Shiyu and Taghavi, Amirhossein and Wang, Tenghui and Meyer, Samantha M. and Childs-Disney, Jessica L. and Li, Chenglong and Disney, Mattew D. and Li, Yanjun},
	title = {Small Molecule Approach to RNA Targeting Binder Discovery (SMARTBind) Using Deep Learning Without Structural Input},
	year = {2025},
	doi = {10.1101/2025.09.24.678312},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```
