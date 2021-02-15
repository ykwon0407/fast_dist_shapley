#!/bin/bash
#
#SBATCH --job-name=[large_reg_time_100_30_49]
#SBATCH -p owners,mrivas,jamesz,normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yckwon@stanford.edu

unset XDG_RUNTIME_DIR
export PATH='/oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/bin:$PATH'
export PATH='/oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/condabin:$PATH'

source /oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/etc/profile.d/conda.sh
conda activate ml_py36

python3 /oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/main_time_regression.py --sample_size 100 --run_id 49 --dimension 30 --is_DShapley

