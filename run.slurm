#!/bin/bash
#SBATCH --job-name=multi_sync
#SBATCH --account=cds
#SBATCH --open-mode=append
#SBATCH --output=./slurm_out/%j_%x.out
#SBATCH --error=./slurm_out/%j_%x.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate general

python3 ./main.py
"
