import xarray as xr
from glob import glob
import numpy as np
import argparse

import data_util as du


def store_all_day_means(
    exp_name,
    start_year,
    end_year,
    variables,
    base_path,
    outpath,
    monthly_means=True,
    tile=3,
    gfdl_processor=None
):
    mean_type, std_type = get_mean_std_types(monthly_means)

    for subset, vars in variables.items():
        for var in vars:
            var = append_tile_if_cubic(var, subset, tile)

            print(
                f"Storing {subset}.{var} {mean_type} for {exp_name}, "
                f"{start_year}-{end_year}",
                flush=True
            )

            data = load_data(
                exp_name,
                base_path,
                start_year,
                end_year,
                var,
                subset,
                gfdl_processor=gfdl_processor
            )

            data = process_special_variables(data, var)

            data_mean, data_std = calculate_statistics(data, monthly_means)
            data_mean = du.lon_180_to_360(data_mean)
            data_std = du.lon_180_to_360(data_std)
            save_statistics(
                data_mean,
                data_std,
                outpath,
                exp_name,
                mean_type,
                std_type,
                start_year,
                end_year,
                var, 
                subset
            )


def get_mean_std_types(monthly_means):
    return ('monthly_mean', 'monthly_std') if monthly_means else ('mean', 'std')


def get_frequency(var):
    monthly_vars = [
        'melt', 'melts', 'transp', 'FWS', 'FWSv',
        'LWS', 'LWSv', 'evap_land', 'precip'
    ]
    return 'monthly' if var in monthly_vars else 'daily'


def append_tile_if_cubic(var, subset, tile):
    return f"{var}.tile{tile}" if 'cubic' in subset else var


def load_data(exp_name, base_path, start_year, end_year, var, subset, gfdl_processor=None):
    if exp_name == 'c192_obs':
        if var == 'ar_precip':
            ar_analysis = True
        else:
            ar_analysis = False
        return load_obs_data(base_path, start_year, end_year, var, subset, ar_analysis)
    elif var == 'ar_precip':
        return load_ar_precip_data(
            base_path, start_year, end_year, exp_name, gfdl_processor
        )
    else:
        return load_model_data(
            base_path, exp_name, gfdl_processor, subset, var, start_year, end_year
        )


def load_obs_data(base_path, start_year, end_year, var, subset, ar_analysis):
    data = xr.concat([
        du.load_obs_data(
            base_path, year, era5_variables=None, 
            exp_name='c192_obs', ar_analysis=ar_analysis, obs_pr_dataset=subset,
            min_pr_threshold=1/86400,
            low_pr_value=0,
            min_pr_var='precip',
            ar_masked_vars=['precip'])
        for year in range(start_year, end_year + 1)
    ], dim='time')
    return data


def load_ar_precip_data(base_path, start_year, end_year, exp_name, gfdl_processor):
    return xr.concat([
        du.load_model_data(
            base_path,
            year,
            ['precip'],
            exp_name,
            ar_analysis=True,
            min_pr_threshold=1/86400,
            min_pr_var='precip',
            low_pr_value=0,
            gfdl_processor=gfdl_processor
        )
        for year in range(start_year, end_year + 1)
    ], dim='time')


def load_model_data(base_path, exp_name, gfdl_processor, subset, var, start_year, end_year):
    paths = [
        p for p in glob(
            f'{base_path}{exp_name}/{gfdl_processor}/pp/{subset}/ts/'
            f'{get_frequency(var)}/1yr/{subset}.*-*.{var}.nc'
        )
        if np.any([f'.{year}' in p for year in np.arange(start_year, end_year + 1)])
    ]
    return xr.open_mfdataset(paths)


def process_special_variables(data, var):
    if var == 'pr':
        data['pr_intensity'] = data.pr.where(data.pr * 86400 > 1)
    elif var == 'prsn':
        data['prsn_intensity'] = data.prsn.where(data.prsn * 86400 > 1)
    return data


def calculate_statistics(data, monthly_means):
    drop_vars = ['average_DT', 'average_T1', 
                 'average_T2', 'time_bnds', 'ar_shape']
    if monthly_means: 
        data_grouped = data.drop_vars(drop_vars, errors='ignore').groupby('time.month')
        return data_grouped.mean(), data_grouped.std()
    else:
        return data.mean('time'), data.std('time')


def save_statistics(data_mean, data_std, outpath, exp_name, mean_type, std_type, start_year, end_year, var, subset):
    data_mean.to_netcdf(
        f'{outpath}/{exp_name}/{exp_name}_all_day_{mean_type}'
        f'.{start_year}-{end_year}.{subset}.{var}.nc' if exp_name == 'c192_obs' else f'.{start_year}-{end_year}.{var}.nc'
    )
    data_std.to_netcdf(
        f'{outpath}/{exp_name}/{exp_name}_all_day_{std_type}'
        f'.{start_year}-{end_year}.{subset}.{var}.nc' if exp_name == 'c192_obs' else f'.{start_year}-{end_year}.{var}.nc'
    )

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min')
    parser.add_argument("--subset", type=str, default='land')
    args = parser.parse_args() 
    exp_name = args.exp_name
    subset = args.subset
    base_path = '/archive/Ming.Zhao/awg/2023.04/'
    gfdl_processor = 'gfdl.ncrc5-intel23-classic-prod-openmp'
    variables = {
        # 'c192_obs': ['prw', 'pr'],
        # 'atmos_cmip': ['pr', 'prsn']
        # 'atmos_cmip': ['ts', 'prw', 'pr', 'prsn', 'wap500'],
        # 'land_cmip': ['mrro', 'mrsos', 'mrso', 'snw'], 
        # # 'atmos_cubic': ['']
        'land': ['ar_precip'],
        # 'land_cubic': ['FWS', 'FWSv', 'LWS', 'LWSv', 
        #                'precip', 'evap_land'],
        # 'river': ['rv_d_h2o', 'rv_o_h2o'],
        # 'river_cubic': ['rv_d_h2o', 'rv_o_h2o'],
        # 'imerg': ['ar_precip'],
                 }
    outpath = '/archive/Marc.Prange/all_day_means/'
    start_year = 1980
    end_year = 2019
    store_all_day_means(
        exp_name, start_year, end_year,
        variables,
        base_path, outpath, monthly_means=True, tile=5, gfdl_processor=gfdl_processor)
    
if __name__ == '__main__':
    _main()
