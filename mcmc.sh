#!/bin/bash
#SBATCH -J mcmc
#SBATCH --partition=koes_gpu
#SBATCH --constraint L40
#SBATCH -o slurm_output/tests/%A_%a.out
#SBATCH -e slurm_output/tests/%A_%a.out
#SBATCH --gres=gpu:1
##SBATCH --array=1-20
##SBATCH -C M16|M20|M24|M40|M46
##SBATCH -C C7|C8
# #SBATCH --exclude=g001,g102,g104,g001,g013,g011,g019
# SBATCH --nodelist g005
# #SBATCH -C M16|M20|M24
# #SBATCH -C M12|M16|M20|M24
# #SBATCH --nodelist g122
# #SBATCH --mem=16GB

hostname

source ~/.bashrc
source activate rebyob

#cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" job_list.txt `
#python mcmc.py --energy lj55 --n_chains 12800 --n_steps 500000 --step_size 0.0075 --run_name lj55_long --box_size 2.0 --max_step 0.01 --plot_frequency 1000 --save_endpoint True --endpoint_name lj55_long_3.npy --checkpoint lj55_long_2.npy
python dem/train.py experiment=lj55_idem

eval $cmd
exit
