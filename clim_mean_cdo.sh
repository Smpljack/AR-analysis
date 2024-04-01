#!/bin/bash

exp_name="c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020"
obs_exp_name="c192_obs"
outpath="/archive/Marc.Prange/clim_means/${exp_name}/"
obs_outpath="/archive/Marc.Prange/clim_means/${obs_exp_name}/"

freq="daily"
base_path="/archive/Ming.Zhao/awg/2022.03/${exp_name}/gfdl.ncrc4-intel-prod-openmp/pp/"
land_path="${base_path}land/ts/${freq}/1yr/"
land_cmip_path="${base_path}land_cmip/ts/${freq}/1yr/"
atmos_cmip_path="${base_path}atmos_cmip/ts/${freq}/1yr/"
atmos_cmip_6hr_path="${base_path}atmos_cmip/ts/6hr/1yr/"
river_path="${base_path}river/ts/${freq}/1yr/"
obs_path="/archive/Ming.Zhao/awg/2022.03/${obs_exp_name}/atmos_data/daily_era5/"

land_vars=("vegn_temp_min" "vegn_temp_max" "t_ref_min" "t_ref_max" "soil_liq" "soil_ice" "snow_soil" "snow_lake" "precip" "npp evap_land")
land_cmip_vars=("mrro" "mrso" "mrsos" "snw" "snc")
atmos_cmip_vars=("ts" "prw" "pr" "prsn" "wap500")
atmos_cmip_6hr_vars=("ivtx" "ivty")
river_vars=("rv_o_h2o" "rv_d_h2o")
obs_vars=("prw" "ivtx" "ivty")


# for var in ${atmos_cmip_vars[@]} ; do
#     temppath=$(mktemp -d)
#     cdo cat ${atmos_cmip_path}atmos_cmip.{1980..2019}0101-*1231.${var}.nc ${temppath}/data.nc
#     cdo -timmean ${temppath}/data.nc ${outpath}${exp_name}_1980-2019_mean.${var}.nc
#     rm -rf ${temppath}
# done

for var in ${atmos_cmip_6hr_vars[@]} ; do
    temppath=$(mktemp -d)
    cdo cat ${atmos_cmip_6hr_path}atmos_cmip.{1980..2019}010100-*123123.${var}.nc ${temppath}/data.nc
    cdo -timmean ${temppath}/data.nc ${outpath}${exp_name}_1980-2019_mean.${var}.nc
    rm -rf ${temppath}
done

for var in ${land_cmip_vars[@]} ; do
    temppath=$(mktemp -d)
    cdo cat ${land_cmip_path}land_cmip.{1980..2019}0101-*1231.${var}.nc ${temppath}/data.nc
    cdo -timmean ${temppath}/data.nc ${outpath}${exp_name}_1980-2019_mean.${var}.nc
    rm -rf ${temppath}
done

for var in ${land_vars[@]} ; do
    temppath=$(mktemp -d)
    cdo cat ${land_path}land.{1980..2019}0101-*1231.${var}.nc ${temppath}/data.nc
    cdo -timmean ${temppath}/data.nc ${outpath}${exp_name}_1980-2019_mean.${var}.nc
    rm -rf ${temppath}
done

for var in ${river_vars[@]} ; do
    temppath=$(mktemp -d)
    cdo cat ${river_path}river.{1980..2019}0101-*1231.${var}.nc ${temppath}/data.nc
    cdo -timmean ${temppath}/data.nc ${outpath}${exp_name}_1980-2019_mean.${var}.nc
    rm -rf ${temppath}
done

for var in ${obs_vars[@]} ; do
    temppath=$(mktemp -d)
    cdo cat ${obs_path}ERA5.{1980..2019}0101-*1231.${var}.nc ${temppath}/data.nc
    cdo -timmean ${temppath}/data.nc ${obs_outpath}${obs_exp_name}_1980-2019_mean.${var}.nc
    rm -rf ${temppath}
done