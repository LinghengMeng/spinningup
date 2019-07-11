#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=1-04:00            # time (DD-HH:MM)
#SBATCH --output=./job_scripts_output/dbedpg_RoboschoolAnt-v1_0_20_1_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
module load cuda cudnn
source ~/tf_gpu/bin/activate
python ./dbedpg_distributed.py  --env RoboschoolAnt-v1 --seed 0 --ensemble_size 20 --replay_buf_bootstrap_p 1 --exp_name dbedpg_distributed_RoboschoolAnt-v1_0_20