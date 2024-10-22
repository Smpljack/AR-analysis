#!/bin/bash
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --account=gfdl_w
#SBATCH --job-name=gpm_daily_c192
#SBATCH --chdir=.
#SBATCH -o /home/Marc.Prange/work/AR-analysis/logs/%x.o%j
#SBATCH -e /home/Marc.Prange/work/AR-analysis/logs/%x.e%j
module load cdo

gridfile="/archive/Ming.Zhao/awg/2022.03/c192_obs/atmos_data/daily_mswep/mswep.19810101-19811231.precipitation.nc"
for year in {2015..2020}
do
    infile="/archive/wnd/GPM/combined/3hr/GPM_${year}.nc"
    outfile="/archive/Marc.Prange/GPM/daily/GPM_${year}.c192_daily.nc"
    cdo -remapbil,${gridfile} -daymean ${infile} ${outfile}
done