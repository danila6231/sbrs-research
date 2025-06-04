#!/bin/bash

#SBATCH --account=cs453_553    ### Account used for job submission
#SBATCH --partition=gpu       ### Similar to a queue in PBS
#SBATCH --job-name=GPUjob     ### Job Name
#SBATCH --time=1-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gpus=1              ### General REServation of gpu:number of gpus
#SBATCH --mem=8G              ### General REServation of gpu:number of gpus
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=danila@uoregon.edu

source .venv/bin/activate
python3 main.py