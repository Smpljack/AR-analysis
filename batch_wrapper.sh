#!/bin/bash -l
#

#BATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --account=gfdl_w
#SBATCH --job-name=ar_day_stat
#SBATCH --chdir=.
#SBATCH -o /home/Marc.Prange/work/AR-analysis/logs/%x.o%j
#SBATCH -e /home/Marc.Prange/work/AR-analysis/logs/%x.e%j

module load conda
conda activate work_python

python data_util.py --exp_name=$1 #run_multiprocessing.py --lon_min=$1 --lon_max=$2 --exp_name=$3
