#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=24:00:00
# set name of job
#SBATCH --job-name=jobzl432
# set number of GPUs
#SBATCH --gres=gpu:1 -p small
# set RAM size
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=zl432@cantab.ac.uk
module load pytorch/1.12.1
source activate epi
cd /jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM
wandb offline

# python test_finetune.py --dataset='clinvar_CLNSIG_dnabert2' --epochs=100 --gpus=8 --num_workers=16
# python test_finetune_delta.py --dataset='cellpassport_hyena-tiny' --epochs=100 --gpus=1 --num_workers=12
# python test_clivar.py
# srun python temp_sanity_check_species_list.py
# python test_finetune.py --dataset='cellpassport_hyena-tiny' --epochs=100 --gpus=1 --num_workers=6
##python test_finetune_delta.py --dataset='sqtl_slope_dnabert2' --epochs=100 --gpus=1 --num_workers=4 --mode=test --checkpoint=/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/Genomic-FM/llm1arxd/checkpoints/epoch=99-step=31800.ckpt
# python test_finetune_delta.py --dataset='cellpassport_hyena-tiny' --epochs=100 --gpus=1 --num_workers=4
# python test_finetune_delta.py --dataset='sqlt_splice-change_hyena-tiny' --epochs=100 --gpus=1 --num_workers=4
# python test_finetune_delta.py --dataset='sqtl_slope_hyena-tiny' --epochs=50 --gpus=1 --num_workers=4
# python test_finetune_delta.py --dataset='sqtl_slope_hyena-tiny' --epochs=50 --gpus=1 --num_workers=4
# python test_finetune_delta.py --dataset='sqtl_slope_hyena-tiny' --epochs=50 --gpus=1 --num_workers=4
# python test_finetune_delta.py --dataset='oligogenic_hyena-tiny' --epochs=50 --gpus=1 --num_workers=4
# python test_finetune_fullsize.py --dataset='sqtl_slope_hyena-tiny' --epochs=50 --gpus=1 --num_workers=4


##########################
## Run Hyena vs. Six Tasks: CliVar_CLNSIG, ClinVar_CLNREVSTAT,CellPassport,GeneKO,sQTLs,eQTLs
##########################
# python test_finetune_delta.py --dataset='clinvar_CLNDN_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_hyena.yaml
# python test_finetune_delta.py --dataset='sqtl_pval_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_hyena.yaml
# python test_finetune_delta.py --dataset='geneko-single_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_hyena.yaml
# srun python test_finetune_delta.py --dataset='clinvar_CLNSIG_hyena-tiny' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_hyena.yaml --seed=0



##########################
## Run DNABERT2 vs. Six Tasks: CliVar_CLNSIG, ClinVar_CLNREVSTAT,CellPassport,GeneKO,sQTLs,eQTLs
##########################
python test_finetune_delta.py --dataset='clinvar_CLNDN_dnabert2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_dnabert2.yaml
# python test_finetune_delta.py --dataset='sqtl_pval_dnabert2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_dnabert2.yaml
# python test_finetune_delta.py --dataset='geneko-single_dnabert2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_dnabert2.yaml
# python test_finetune_delta.py --dataset='clinvar_CLNSIG_dnabert2' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_dnabert2.yaml --seed=42
# python test_finetune_delta.py --dataset='clinvar_CLNSIG_dnabert2' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_dnabert2.yaml --seed=0

##########################
## Run nt
##########################

# python test_finetune_delta.py --dataset='clinvar_CLNDN_nt' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_nt.yaml
# python test_finetune_delta.py --dataset='sqtl_pval_nt' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_nt.yaml
# python test_finetune_delta.py --dataset='geneko-single_nt' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_nt.yaml
# srun python test_finetune_delta.py --dataset='clinvar_CLNSIG_nt' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_nt.yaml
# python test_finetune_delta.py --dataset='clinvar_CLNSIG_nt' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_nt.yaml --seed=1


##########################
## Run Hyena Random
##########################
# python test_finetune_delta.py --mode=random --dataset='clinvar_CLNDN_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_hyena.yaml
# python test_finetune_delta.py --mode=random --dataset='sqtl_pval_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_hyena.yaml
# python test_finetune_delta.py --mode=random --dataset='geneko-single_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_hyena.yaml
# python test_finetune_delta.py --mode=random --dataset='clinvar_CLNSIG_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_hyena.yaml



##########################
## Run ntv2
##########################

# python test_finetune_delta.py --dataset='clinvar_CLNDN_ntv2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_ntv2.yaml
# python test_finetune_delta.py --dataset='sqtl_pval_ntv2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_ntv2.yaml
# python test_finetune_delta.py --dataset='geneko-single_ntv2' --epochs=100 --gpus=1 --num_workers=1 --config=configs/finetune_ntv2.yaml
# srun python test_finetune_delta.py --dataset='clinvar_CLNSIG_ntv2' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_ntv2.yaml --seed=1


##########################
# Missing Runs
##########################
# python test_finetune_delta.py --dataset='clinvar_CLNDN_ntv2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_ntv2.yaml
# python test_finetune_delta.py --dataset='clinvar_CLNDN_nt' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_nt.yaml
#### CLNSIG #####

# srun python test_finetune_delta.py --dataset='clinvar_CLNSIG_hyena-tiny' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_hyena.yaml --seed=0
# srun python test_finetune_delta.py --dataset='clinvar_CLNSIG_dnabert2' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_dnabert2.yaml
# srun python test_finetune_delta.py --dataset='clinvar_CLNSIG_nt' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_nt.yaml
# srun python test_finetune_delta.py --dataset='clinvar_CLNSIG_ntv2' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_ntv2.yaml
