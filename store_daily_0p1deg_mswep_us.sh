#!/bin/bash
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --account=gfdl_w
#SBATCH --job-name=mswep_daily_na
#SBATCH --chdir=.
#SBATCH -o /home/Marc.Prange/work/AR-analysis/logs/%x.o%j
#SBATCH -e /home/Marc.Prange/work/AR-analysis/logs/%x.e%j
module load cdo

for year in {1979..2020}
do
    infile="/archive/Wenhao.Dong/MSWEP/MSWEPV2/3hr_year/MSWEPV2_${year}.0.1.nc"
    outfile="/archive/Marc.Prange/na_data/mswep/MSWEPV2_${year}.0.1_daily_na.nc"
    cdo -sellonlatbox,-140,-60,20,70 -daymean ${infile} ${outfile}
done