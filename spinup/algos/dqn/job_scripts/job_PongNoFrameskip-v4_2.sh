#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-05:00            # time (DD-HH:MM)
#SBATCH --output=./job_scripts_output/dqn_PongNoFrameskip-v4_2_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
module load cuda cudnn 
source ~/tf_gpu/bin/activate
python ./dqn.py  -g PongNoFrameskip-v4 -s 2 --l 2 --exp_name dqn_PongNoFrameskip-v4_2