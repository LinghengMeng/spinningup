#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=1-04:00            # time (DD-HH:MM)
#SBATCH --output=./job_scripts_output/dbedpg_RoboschoolHalfCheetah-v1_1_20_0.75_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
module load cuda cudnn
source ~/tf_gpu/bin/activate
python ./dbedpg_distributed.py  --env RoboschoolHalfCheetah-v1 --seed 1 --ensemble_size 20 --replay_buf_bootstrap_p 0.75 --exp_name dbedpg_distributed_RoboschoolHalfCheetah-v1_1_20