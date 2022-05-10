#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu
#SBATCH --job-name=python-cuad
#SBATCH --output=cuad.out

module purge

singularity exec --nv \
	    --overlay /scratch/yt2093/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; bash ./run.sh"
