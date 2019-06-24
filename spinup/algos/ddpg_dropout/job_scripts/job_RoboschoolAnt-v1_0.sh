#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-04:00            # time (DD-HH:MM)
#SBATCH --output=./job_scripts_output/ddpg_dropout_RoboschoolAnt-v1_0_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
module load cuda cudnn
source ~/tf_gpu/bin/activate
python ./ddpg_dropout.py  --env RoboschoolAnt-v1 --new_mlp --dropout_rate 0 --exp_name ddpg_dropout_RoboschoolAnt-v1_0_onlyQDropout