# How to run it on NYU-HPC

## Step1 using singularity overlays to create execute environment
Create a DL directory for environment and project
```
mkdir /scratch/<NetID>/dl
cd /scratch/<NetID>/dl
```

Choose a overlay
```
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-7.5GB-300K.ext3.gz .
gunzip overlay-7.5GB-300K.ext3.gz
```

Launch the appropriate Singularity container and enter it
```
singularity exec --overlay overlay-7.5GB-300K.ext3 /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh

sh Miniconda3-py38_4.9.2-Linux-x86_64.sh -b -p /ext3/miniconda3

```

Create a wrapper script /ext3/env.sh
```
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
```

Activate your conda environment and install packages (Depends on your project)
```
source /ext3/env.sh

conda update -n base conda -y
conda clean --all --yes
conda install pip
conda install ipykernel # Note: ipykernel is required to run as a kernel in the Open OnDemand Jupyter Notebooks

pip install torch==1.7.0

pip install jupyter jupyterhub pandas matplotlib scipy scikit-learn scikit-image Pillow

pip install transformers==3.4.0 tensorboardX apex tensorflow_datasets ptvsd
```

Exit the Singularity container and Test
```
exit
mv overlay-7.5GB-300K.ext3 my_pytorch.ext3

singularity exec --overlay /scratch/<NetID>/dl/my_pytorch.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c 'source /ext3/env.sh; python -c "import torch; print(torch.__file__); print(torch.__version__)"'
```

[Singularity with Miniconda](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)

## Step2 Pull your project
Enter dl directory
```
cd /scratch/<NetID>/dl
git clone git@github.com:Fieldwater/cuad.git

cd cuad
unzip data.zip -d ./data/
```

## Step3 Submit a SLURM batch job

Create a run.sbatch
```
#!/bin/bash

#SBATCH --nodes=3       # requests N compute servers
#SBATCH --ntasks-per-node=2   # runs N tasks on each server
#SBATCH --cpus-per-task=1     # uess N compute core per task
#SBATCH --time=10:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu
#SBATCH --job-name=python-cuad
#SBATCH --output=cuad.out

module purge

singularity exec --nv \
	    --overlay /scratch/<NetID>/dl/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; bash ./run.sh"
```

Submit your job
```
sbatch run.sbatch
```

## Step4 Check job status 
Check in terminal
```
# follow your output dynamically
tail -f cuad.out

# show job description
scontrol show job <id>

# queue job
squeue -u yt2093

# cancel jobs
scancel <id1> <id2>
```

Or you can use [GUI](https://ood.hpc.nyu.edu/pun/sys/dashboard/)
- Jobs -> Active Jobs


# Contract Understanding Atticus Dataset

This repository contains code for the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad), pronounced "kwad", a dataset for legal contract review curated by the Atticus Project. It is part of the associated paper [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268) by [Dan Hendrycks](http://danhendrycks.com/), [Collin Burns](http://collinpburns.com), Anya Chen, and Spencer Ball.

Contract review is a task about "finding needles in a haystack." 
We find that Transformer models have nascent performance on CUAD, but that this performance is strongly influenced by model design and training dataset size. Despite some promising results, there is still substantial room for improvement. As one of the only large, specialized NLP benchmarks annotated by experts, CUAD can serve as a challenging research benchmark for the broader NLP community.

<img align="center" src="contract_review.png" width="1000">

For more details about CUAD and legal contract review, see the [Atticus Project website](https://www.atticusprojectai.org/cuad).

## Trained Models

We [provide checkpoints](https://zenodo.org/record/4599830) for three of the best models fine-tuned on CUAD: RoBERTa-base (~100M parameters), RoBERTa-large (~300M parameters), and DeBERTa-xlarge (~900M parameters). 

## Extra Data
Researchers may be interested in several gigabytes of unlabeled contract pretraining data, which is available [here](https://drive.google.com/file/d/1of37X0hAhECQ3BN_004D8gm6V88tgZaB/view?usp=sharing).

## Requirements

This repository requires the HuggingFace [Transformers](https://huggingface.co/transformers) library. It was tested with Python 3.8, PyTorch 1.7, and Transformers 4.3/4.4. 

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2021cuad,
          title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review}, 
          author={Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
          journal={NeurIPS},
          year={2021}
    }
