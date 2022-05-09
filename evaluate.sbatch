#!/bin/bash

#SBATCH --nodes=1       # requests N compute servers
#SBATCH --ntasks-per-node=1   # runs N tasks on each server
#SBATCH --cpus-per-task=1     # uess N compute core per task
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=python-cuad
#SBATCH --output=evaluate.out

module purge

singularity exec --nv \
	    --overlay /scratch/yt2093/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python ./evaluate.py"

