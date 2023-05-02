#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=infer_avocado
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=A40short
#SBATCH -x node-02
#SBATCH --output=/home/s6kogase/work/avocodo_fork/out/infer_avoco_%j.out
#SBATCH --error=/home/s6kogase/work/avocodo_fork/out/infer_avoco_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py38

python avocodo/inference.py --version 0 --checkpoint_file_id epoch=146-step=238728.ckpt

echo "Goodbye at $(date)."