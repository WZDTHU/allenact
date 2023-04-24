#!/bin/bash
#SBATCH --job-name=train_A2SP    # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=30       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --qos=level1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/home/wangzidong/scratch/work/experiment_output/allenact/slurm_out/slurm-%j.out


module purge
module load slurm/slurm/20.11.8
module load anaconda3/2021.11
module load vulkan/1.3.224.1

source activate A2SP_env
conda deactivate
conda activate A2SP_env 


srun python -u main.py -o /home/wangzidong/scratch/work/experiment_output/allenact/A2SP_out