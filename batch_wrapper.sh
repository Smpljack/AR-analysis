#!/bin/bash -l
#

#BATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=gfdl_w
#SBATCH --job-name=composite_precip_mean
#SBATCH --chdir=.
#SBATCH -o /home/Marc.Prange/work/AR-analysis/logs/%x.o%j
#SBATCH -e /home/Marc.Prange/work/AR-analysis/logs/%x.e%j

module load conda
conda activate /home/Marc.Prange/miniconda3/envs/AR_analysis

python processing.py --year=$1 #run_multiprocessing.py --lon_min=$1 --lon_max=$2 --exp_name=$3
