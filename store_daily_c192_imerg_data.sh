#!/bin/bash
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --account=gfdl_w
#SBATCH --job-name=imerg_daily_c192
#SBATCH --chdir=.
#SBATCH -o /home/Marc.Prange/work/AR-analysis/logs/%x.o%j
#SBATCH -e /home/Marc.Prange/work/AR-analysis/logs/%x.e%j
module load cdo


gridfile="/archive/Ming.Zhao/awg/2022.03/c192_obs/atmos_data/daily_mswep/mswep.19810101-19811231.precipitation.nc"
# for year in {2001..2002}
#     do
year=$1
for month in 07 ; 
    do
        infile="/archive/Marc.Prange/IMERG/temp/IMERG_daymean_raw_res_${year}_${month}.V07B.nc"
        outfile="/archive/Marc.Prange/IMERG/daily/IMERG_${month}_${year}.c192_daily.nc"
        cdo -O -remapbil,${gridfile} ${infile} ${outfile}
done
# done