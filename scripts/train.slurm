#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=train_avocado
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=A40medium
#SBATCH -x node-02,node-06
#SBATCH --output=/home/s6kogase/work/avocodo_fork/out/train_avoco_%j.out
#SBATCH --error=/home/s6kogase/work/avocodo_fork/out/train_avoco_%j.err

source ${HOME}/.bashrc

echo "Hello from job $SLURM_JOB_ID on $(hostname) at $(date)."

conda activate py38

python avocodo/train.py --config avocodo/configs/avocodo_v1.json --num_gpus 1 \
    --resume_checkpoint_path /home/s6kogase/work/avocodo_fork/logs/Avocodo/version_1/checkpoints/epoch=84-step=138040.ckpt

echo "Goodbye at $(date)."