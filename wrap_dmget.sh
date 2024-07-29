#!/bin/bash

freq="daily"
base_path="/archive/Ming.Zhao/awg/2022.03/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K/gfdl.ncrc4-intel-prod-openmp/pp/"
land_path="${base_path}land/ts/${freq}/1yr/"
land_cmip_path="${base_path}land_cmip/ts/${freq}/1yr/"
atmos_cmip_path="${base_path}atmos_cmip/ts/${freq}/1yr/"
atmos_cmip_6hr_path="${base_path}atmos_cmip/ts/6hr/1yr/"
river_path="${base_path}river/ts/${freq}/1yr/"
obs_path="/archive/Ming.Zhao/awg/2022.03/c192_obs/atmos_data/daily_era5/"

land_vars=("vegn_temp_min" "vegn_temp_max" "t_ref_min" "t_ref_max" "soil_liq" "soil_ice" "snow_soil" "snow_lake" "precip" "npp evap_land")
land_cmip_vars=("mrro" "mrso" "mrsos" "snw" "snc")
atmos_cmip_vars=("ts" "prw" "pr" "prsn" "wap500")
atmos_cmip_6hr_vars=("ivtx" "ivty")
river_vars=("rv_o_h2o" "rv_d_h2o")
obs_vars=("prw" "ivtx" "ivty")

# for var in ${obs_vars[@]} ; do
#     dmget ${obs_path}ERA5.{1951..2020}0101-*1231.${var}.nc &
# done

for var in ${land_cmip_vars[@]} ; do
    dmget ${land_cmip_path}land_cmip.{1951..2020}0101-*1231.${var}.nc &
done

for var in ${land_vars[@]} ; do
    dmget ${land_path}land.{1951..2020}0101-*1231.${var}.nc &
done

for var in ${atmos_cmip_vars[@]} ; do
    dmget ${atmos_cmip_path}atmos_cmip.{1951..2020}0101-*1231.${var}.nc &
done

for var in ${atmos_cmip_6hr_vars[@]} ; do
    dmget ${atmos_cmip_6hr_path}atmos_cmip.{1951..2020}010100-*123123.${var}.nc &
done

for var in ${river_vars[@]} ; do
    dmget ${river_path}river.{1951..2020}0101-*1231.${var}.nc &
done



