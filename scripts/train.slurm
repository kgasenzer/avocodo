#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_avocado
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=A40short
#SBATCH -x node-02
#SBATCH --output=/home/s6kogase/work/avocodo/out/train_voco_%j.out
#SBATCH --error=/home/s6kogase/work/avocodo/out/train_voco_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py38

python avocodo/train.py --config avocodo/configs/avocodo_v1.json

echo "Goodbye at $(date)."