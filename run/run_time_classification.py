import os, time

for dimension in [30]:
    for sample_size in [100]:
        for run_id in range(50):
            cmd = f"""#!/bin/bash
#
#SBATCH --job-name=[large_clf_time_{sample_size}_{dimension}_{run_id}]
#SBATCH -p owners,mrivas,jamesz,normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yckwon@stanford.edu

unset XDG_RUNTIME_DIR
export PATH='/oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/bin:$PATH'
export PATH='/oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/condabin:$PATH'

source /oak/stanford/groups/mrivas/users/yckwon/software/anaconda3/etc/profile.d/conda.sh
conda activate ml_py36

python3 /oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/main_time_classification.py --sample_size {sample_size} --run_id {run_id} --dimension {dimension} --is_DShapley 

"""
            print(cmd)
            with open("run_temp.sh", "w") as f:
                f.write(cmd)
            os.system('sbatch run_temp.sh')
            time.sleep(1)


