#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=python-xlnet
#SBATCH --output=xlnet.out

module purge

singularity exec --nv \
	    --overlay /scratch/yt2093/dl/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; bash ./xlnet_run.sh"
