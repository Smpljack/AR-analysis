import xarray as xr
from glob import glob
import numpy as np
import argparse

import data_util as du


def store_all_day_means(
    exp_name, start_year, end_year, variables,
    base_path, outpath, gfdl_processor, monthly_means=True, tile=3):
    if monthly_means:
        mean_type = 'monthly_mean'
        std_type = 'monthly_std'
    else:
        mean_type = 'mean'
        std_type = 'std'
    for subset, vars in variables.items():
        for var in vars:
            if 'cubic' in subset:
                var += f'.tile{tile}'
            print(
                f"Storing {subset}.{var} {mean_type} for {exp_name}, {start_year}-{end_year}", 
                flush=True)
            if var in ['melt', 'melts', 'transp']:
                freq = 'monthly'
            else:
                freq = 'daily'
            paths = [p for p in 
                    glob(f'{base_path}{exp_name}/{gfdl_processor}/pp/{subset}/ts/{freq}/1yr/'
                        f'{subset}.*-*.{var}.nc' ) 
                    if np.any([f'.{year}' in p for year in np.arange(start_year, end_year+1)])]
            data = xr.open_mfdataset(paths)
            if monthly_means:
                data_grouped = data.drop_vars(
                    ['average_DT', 'average_T1', 'average_T2', 'time_bnds'],
                    errors='ignore'
                    ).groupby('time.month')
                data_mean = data_grouped.mean()
                data_std = data_grouped.std()
                
            else:
                data_mean = data.mean('time')
                data_std = data.std('time')
            data_mean.to_netcdf(
                f'{outpath}/{exp_name}/{exp_name}_all_day_{mean_type}'
                f'.{start_year}-{end_year}.{var}.nc'
            )
            data_std.to_netcdf(
                f'{outpath}/{exp_name}/{exp_name}_all_day_{std_type}'
                f'.{start_year}-{end_year}.{var}.nc'
            )
            

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K')
    args = parser.parse_args() 
    exp_name = args.exp_name
    base_path = '/archive/Ming.Zhao/awg/2023.04/'
    gfdl_processor = 'gfdl.ncrc5-intel23-classic-prod-openmp'
    variables = {
        'atmos_cmip': ['ts', 'prw', 'pr', 'prsn', 'wap500'],
        'land_cmip': ['mrro', 'mrsos', 'mrso', 'snw'], 
        # 'atmos_cubic': ['']
        'land': ['transp', 'melt', 'melts'],
        # 'land_cubic': ['transp', 'melt', 'melts'],
        # 'river': ['rv_d_h2o', 'rv_o_h2o'],
        'river_cubic': ['rv_d_h2o', 'rv_o_h2o'],
                 }
    outpath = '/archive/Marc.Prange/all_day_means/'
    start_year = 1980
    end_year = 2019
    store_all_day_means(
        exp_name, start_year, end_year, 
        variables,
        base_path, outpath, gfdl_processor, monthly_means=True, tile=3)
    
if __name__ == '__main__':
    _main()