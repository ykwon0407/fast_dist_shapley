import os, time

for dataset in ['diabetes_scale']:
    for run_id in range(50):
        cmd = f"""#!/bin/bash
#
#SBATCH --job-name=[DSV_density_{dataset}_{run_id}]
#SBATCH -p owners,mrivas,jamesz,normal
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yckwon@stanford.edu
#SBATCH --output=/oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/temp3/density/{dataset}/slurm-%j.out

unset XDG_RUNTIME_DIR
export PATH='/oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/bin:$PATH'
export PATH='/oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/condabin:$PATH'

source /oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/etc/profile.d/conda.sh
conda activate ml_py36

python3 /oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/main_density.py --dataset {dataset} --run_id {run_id} 

"""
        print(cmd)
        with open("run_temp.sh", "w") as f:
            f.write(cmd)
        os.system('sbatch run_temp.sh')
        time.sleep(1)



