#!/bin/bash

exp_name="c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K"
inpath="/archive/Ming.Zhao/awg/2022.03/${exp_name}/gfdl.ncrc4-intel-prod-openmp/pp/atmos_cmip/ts/daily/1yr/"
outpath="/archive/Marc.Prange/clim_means/${exp_name}/"

for var in wap500 ; do
    temppath=$(mktemp -d)
    cdo cat ${inpath}atmos_cmip.{1990..2020}0101-*1231.${var}.nc ${temppath}/data.nc
    cdo -timmean ${temppath}/data.nc ${outpath}${exp_name}_1990-2020_mean.${var}.nc
    rm -rf ${temppath}
done
wait

