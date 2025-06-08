#!/bin/bash

#SBATCH --account=cs453_553    ### Account used for job submission
#SBATCH --partition=gpu       ### Similar to a queue in PBS
#SBATCH --job-name=GPUjob     ### Job Name
#SBATCH --output=./results/ta-feng_base-benchmark-small-seq-len.out
#SBATCH --time=1-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gpus=1              ### General REServation of gpu:number of gpus
#SBATCH --mem=4G              ### General REServation of gpu:number of gpus
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=danila@uoregon.edu

source .venv/bin/activate
python3 run_base.py --MAX_ITEM_LIST_LENGTH=5

#--selected_user_features="['age_group']"
#--selected_user_features="['gender', 'occupation']"
#--feature_emb_hidden_size=64