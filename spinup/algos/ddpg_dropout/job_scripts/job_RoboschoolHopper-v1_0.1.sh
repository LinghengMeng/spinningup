#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-04:00            # time (DD-HH:MM)
#SBATCH --output=./job_scripts_output/dbedpg_RoboschoolHopper-v1_0.1_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
module load cuda cudnn
source ~/tf_gpu/bin/activate
python ./dbedpg.py  --env RoboschoolHopper-v1 --dropout_rate 0.1 --exp_name dbedpg_RoboschoolHopper-v1_0.1