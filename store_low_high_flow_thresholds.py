import xarray as xr
import numpy as np
import argparse
from glob import glob

import data_util as du


def store_low_high_flow_thresholds(
    exp_name, base_path, gfdl_processor, out_path, start_year, end_year, percentiles=[0.1, 0.9], variable='rv_o_h2o'):
    discharge_long = xr.concat(
    [
        du.load_model_data(
            base_path=base_path,
            year=year,
            variables=['pr', 'prsn', 'rv_o_h2o'],
            exp_name=exp_name,
            gfdl_processor=gfdl_processor,
            ar_analysis=False) 
        for year in range(start_year, end_year+1)
        ], dim='time')
    low_flow = discharge_long[variable].quantile(percentiles[0], dim='time')
    high_flow = discharge_long[variable].quantile(percentiles[1], dim='time')
    if variable == 'rv_o_h2o':
        var_str = 'flow'
    elif variable == 'pr':
        var_str = 'precip'
    # _{str((1-percentiles[0])*365.25).replace(".", "p")}_year-1
    high_flow.to_netcdf(
        f'{out_path}/high_{var_str}_threshold_{exp_name}_{start_year}_{end_year}.nc'
    )
    low_flow.to_netcdf(
        f'{out_path}/low_{var_str}_threshold_{exp_name}_{start_year}_{end_year}.nc'
    )

def store_low_high_flow_statistics(
        flow_threshold, threshold_kind, exp_name, base_path, gfdl_processor, out_path, start_year, end_year,
        variable='rv_o_h2o'):
    for year in range(start_year, end_year+1):
        daily_data = du.load_model_data(
                base_path=base_path,
                year=year,
                variables=['pr', 'prsn', 'rv_o_h2o'],
                exp_name=exp_name,
                gfdl_processor=gfdl_processor,
                ar_analysis=False)
        if threshold_kind == 'high':
            flow_extreme_bool = daily_data[variable] > flow_threshold
        if threshold_kind == 'low':
            flow_extreme_bool = daily_data[variable] < flow_threshold
        monthly_flow_extreme_count = flow_extreme_bool.groupby('time.month').sum()
        monthly_flow_extreme_mean = daily_data[variable].where(flow_extreme_bool).groupby('time.month').mean()
        monthly_flow_extreme_std = daily_data[variable].where(flow_extreme_bool).groupby('time.month').std()
        if variable == 'rv_o_h2o':
            var_str = 'flow'
        elif variable == 'pr':
            var_str = 'precip'
        monthly_flow_extreme_count.to_netcdf(
           f'{out_path}/{threshold_kind}_{var_str}_count_monthly_{exp_name}_{year}.nc'
        )
        monthly_flow_extreme_mean.to_netcdf(
           f'{out_path}/{threshold_kind}_{var_str}_mean_monthly_{exp_name}_{year}.nc'
        )
        monthly_flow_extreme_std.to_netcdf(
           f'{out_path}/{threshold_kind}_{var_str}_std_monthly_{exp_name}_{year}.nc'
        )

def store_mean_low_high_flow_statistics(
        threshold_kind, exp_name, in_path, out_path, start_year, end_year, variable='flow'):
    for stat in ['count', 'mean', 'std']:
        paths = [p for p in glob(f'{in_path}/{threshold_kind}_{variable}_{stat}_monthly_{exp_name}_*.nc')
                if int(p[-7:-3]) in range(start_year, end_year+1)]
        monthly_flow_extreme_stat = xr.open_mfdataset(
            paths, concat_dim='year', combine='nested').to_array().mean('year').squeeze()
        if stat == 'std':
            monthly_flow_extreme_stat.rename(f'{threshold_kind}_{variable}_mean').to_netcdf(
            f'{out_path}/{exp_name}_all_day_monthly_std.'
            f'{start_year}-{end_year}.{threshold_kind}_{variable}_mean.nc'
        )
        else:
            monthly_flow_extreme_stat.rename(f'{threshold_kind}_{variable}_{stat}').to_netcdf(
                f'{out_path}/{exp_name}_all_day_monthly_mean.'
                f'{start_year}-{end_year}.{threshold_kind}_{variable}_{stat}.nc'
            )
    
    

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min')
    args = parser.parse_args()
    exp_name = args.exp_name
    base_path = '/archive/Ming.Zhao/awg/2023.04/'
    gfdl_processor = 'gfdl.ncrc5-intel23-classic-prod-openmp'
    in_path = f'/archive/Marc.Prange/discharge_statistics/{exp_name}'
    out_path = f'/archive/Marc.Prange/discharge_statistics/{exp_name}'
    start_year = 1980
    end_year = 2019
    # store_low_high_flow_thresholds(
    #     exp_name, base_path, gfdl_processor, out_path, start_year, end_year, percentiles=[0.1, 0.9], 
    #     variable='rv_o_h2o')
    
    low_flow_threshold = xr.open_dataarray(
        f'/archive/Marc.Prange/discharge_statistics/c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/'
        f'low_flow_threshold_c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_{start_year}_{end_year}.nc'
    )
    high_flow_threshold = xr.open_dataarray(
        f'/archive/Marc.Prange/discharge_statistics/c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/'
        f'high_flow_threshold_c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_{start_year}_{end_year}.nc' 
    )
    store_low_high_flow_statistics(
        low_flow_threshold, 'low', exp_name, base_path, gfdl_processor, out_path, start_year, end_year,
        variable='rv_o_h2o')
    store_low_high_flow_statistics(
        high_flow_threshold, 'high', exp_name, base_path, gfdl_processor, out_path, start_year, end_year, 
        variable='rv_o_h2o')
    

if __name__ == '__main__':
    _main()