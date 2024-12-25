## Command to fine tune

## Requirements

* python 3.10
* pytorch 2.2.0

conda create -n py310 python=3.10
conda install conda-forge::transformers pytorch::pytorch=2.2.0
conda install bioconda::pyliftover
pip3 install mavehgvs
pip install git+https://github.com/dermatologist/genomic-tokenizer.git@feature/handle-stop-1
conda install bioconda::vcfpy
conda install bioconda::kipoiseq
conda install conda-forge::einops
conda install conda-forge::rich
conda install conda-forge::boto3
conda install conda-forge::scikit-learn
conda install conda-forge::scipy=1.12.0
conda install conda-forge::wandb
conda install conda-forge::pytorch-lightning=1.9.3

*

conda install conda-forge::pytorch-cpu=2.1.2
conda install conda-forge::pandas=2.1.4
sudo apt-get install libgfortran5
sudo apt install -y g++-11


conda install bioconda::vcfpy conda-forge::pytorch-gpu=2.1.2 pytorch::pytorch-cuda=12.4 conda-forge::pytorch-lightning=1.9.3
conda install conda-forge::transformers conda-forge::wandb
conda install conda-forge::scikit-learn conda-forge::rich conda-forge::boto3 conda-forge::einops
conda install bioconda::kipoiseq bioconda::pyliftover
pip3 install mavehgvs
pip install git+https://github.com/dermatologist/genomic-tokenizer.git@feature/handle-stop-1



```
wandb offline # if GPU compute cannot access the internet
python finetune.py --dataset='oligogenic_codon_bert' --epochs=3 --gpus=3 --num_workers=2 --config=configs/finetune_codonbert.yaml --seed=0 --project='Codon-Bert-Olida'
```

## Nohup

```
nohup command </dev/null >/dev/null 2>&1 & # completely detached from terminal

```