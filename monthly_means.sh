#!/bin/bash
module load cdo

# Define input file pattern (assuming annual files with daily data)
exp='c192_obs'
path="/archive/Ming.Zhao/awg/2022.03/c192_obs/atmos_data/daily_era5/"
out_path="/archive/Marc.Prange/era5_monthly/"

for var in 'u_ref' 'v_ref' 'prw' 'ivtx' 'ivty'; do
for (( year=1980; year<=2019; year++ )); do
    infile="${path}ERA5.${year}*.${var}.nc"
    outfile="${out_path}era5_monmean_${year}.${var}.nc"
    echo $infile
    echo $outfile
    # Calculate monthly means using CDO
    cdo -s -O -L -monmean $infile $outfile
done
done