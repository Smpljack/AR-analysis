import xarray as xr
import numpy as np
import argparse
from glob import glob

import data_util as du


def store_low_high_flow_thresholds(
    exp_name,
    base_path,
    gfdl_processor,
    out_path,
    start_year,
    end_year,
    percentiles=[0.1, 0.9],
    variable='rv_o_h2o'
):
    discharge_long = xr.concat(
        [
            load_daily_data(base_path, year, variable, exp_name, gfdl_processor, False)
            for year in range(start_year, end_year + 1)
        ],
        dim='time'
    )
    for pctl in percentiles:
        pctl_value = discharge_long[variable].quantile(pctl, dim='time')
        var_str = get_variable_string(variable, pctl)
        save_threshold(
            pctl_value, out_path, var_str, exp_name, start_year, end_year
        )


def save_threshold(pctl_value, out_path, var_str, exp_name, start_year, end_year):
    filename = (
        f'{out_path}/{var_str}_threshold_{exp_name}_{start_year}_{end_year}.nc'
    )
    pctl_value.to_netcdf(filename)


def store_low_high_flow_statistics(
    flow_thresholds,
    threshold_kinds,
    pctls,
    exp_name,
    base_path,
    gfdl_processor,
    out_path,
    start_year,
    end_year,
    variable='rv_o_h2o',
    ar_condition=False  # Parameter to enable AR condition
):
    for year in range(start_year, end_year + 1):
        daily_data = load_daily_data(
            base_path, year, variable, exp_name, gfdl_processor, ar_condition
        )
        if ar_condition:
            out_variable = 'ar_'+variable
        else:
            out_variable = variable
        for flow_threshold, threshold_kind, pctl in zip(
            flow_thresholds, threshold_kinds, pctls):
            flow_extreme_bool = calculate_flow_extreme(
                daily_data, out_variable, threshold_kind, flow_threshold
            )
            monthly_statistics = calculate_monthly_statistics(
                daily_data, out_variable, flow_extreme_bool
            )
            var_str = get_variable_string(out_variable, pctl)
            save_monthly_statistics(
                monthly_statistics, out_path, threshold_kind, var_str, exp_name, year
            )


def load_daily_data(
    base_path, year, variable, exp_name, gfdl_processor, ar_condition
):
    return du.load_model_data(
        base_path=base_path,
        year=year,
        variables=[variable],
        exp_name=exp_name,
        gfdl_processor=gfdl_processor,
        ar_analysis=ar_condition,
        ar_masked_vars=[variable] if ar_condition else []
    )


def calculate_flow_extreme(daily_data, variable, threshold_kind, flow_threshold):
    if threshold_kind == 'high':
        flow_extreme = daily_data[variable] > flow_threshold
    elif threshold_kind == 'low':
        flow_extreme = daily_data[variable] < flow_threshold
    else:
        raise ValueError("threshold_kind must be 'high' or 'low'")

    return flow_extreme


def calculate_monthly_statistics(daily_data, variable, flow_extreme_bool):
    return {
        'count': flow_extreme_bool.groupby('time.month').sum(),
        'mean': daily_data[variable].where(flow_extreme_bool).groupby('time.month').mean(),
        'std': daily_data[variable].where(flow_extreme_bool).groupby('time.month').std()
    }


def get_variable_string(variable, pctl):
    if variable == 'rv_o_h2o':
        var_str = 'flow'
    elif variable == 'ar_rv_o_h2o':
        var_str = 'ar_flow'
    elif variable == 'precip':
        var_str = 'precip'
    return f'{var_str}_{str(round(calculate_days_per_year(pctl), 2)).replace(".", "p")}_year-1'


def save_monthly_statistics(
    monthly_statistics, out_path, threshold_kind, var_str, exp_name, year
):
    for stat_name, stat_data in monthly_statistics.items():
        filename = (
            f'{out_path}/{threshold_kind}_{var_str}_'
            f'{stat_name}_monthly_{exp_name}_{year}.nc'
        )
        stat_data.to_netcdf(filename)


def store_mean_low_high_flow_statistics(
    threshold_kind,
    exp_name,
    in_path,
    out_path,
    start_year,
    end_year,
    variable='flow'
):
    for stat in ['count', 'mean', 'std']:
        paths = [
            p for p in glob(
                f'{in_path}/{threshold_kind}_{variable}_{stat}_monthly_'
                f'{exp_name}_*.nc'
            )
            if start_year <= int(p[-7:-3]) <= end_year
        ]

        monthly_flow_extreme_stat = (
            xr.open_mfdataset(paths, concat_dim='year', combine='nested')
            .to_array()
            .mean('year')
            .squeeze()
        )

        output_filename = (
            f'{out_path}/{exp_name}_all_day_monthly_'
            f'{"std" if stat == "std" else "mean"}.'
            f'{start_year}-{end_year}.'
            f'{threshold_kind}_{variable}_'
            f'{"mean" if stat == "std" else stat}.nc'
        )

        new_name = f'{threshold_kind}_{variable}_{"mean" if stat == "std" else stat}'
        monthly_flow_extreme_stat.rename(new_name).to_netcdf(output_filename)


def calculate_percentile(days_per_year=1):
    """
    Calculate percentile based on occurrence days per year.

    Args:
    days_per_year (float): Number of occurrence days per year

    Returns:
    float: Calculated percentile
    """
    return 1 - days_per_year / 365.25


def calculate_days_per_year(percentile):
    """
    Calculate the number of occurrence days per year based on percentile.

    Args:
    percentile (float): Percentile value between 0 and 1

    Returns:
    float: Calculated number of days per year
    """
    return 365.25 * (1 - percentile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K'
    )
    args = parser.parse_args()
    exp_name = args.exp_name
    base_path = '/archive/Ming.Zhao/awg/2023.04/'
    gfdl_processor = 'gfdl.ncrc5-intel23-classic-prod-openmp'
    in_path = f'/archive/Marc.Prange/discharge_statistics/{exp_name}'
    out_path = f'/archive/Marc.Prange/all_day_means/{exp_name}'
    start_year = 1980
    end_year = 2019

    # Uncomment and adjust these lines as needed
    # store_low_high_flow_thresholds(
    #     exp_name, base_path, gfdl_processor, out_path, start_year, end_year,
    #     percentiles=[0.1, 0.9], variable='rv_o_h2o'
    # )

    # low_flow_threshold = xr.open_dataarray(
    #     f'/archive/Marc.Prange/discharge_statistics/'
    #     f'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/'
    #     f'low_flow_threshold_c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_'
    #     f'{start_year}_{end_year}.nc'
    # )

    
    threshold_kind_list = ['low'] 
    days_per_year_list = ['328p73']
    prctl_list = [0.1]
    ar_condition = False
    flow_threshold_list = [xr.open_dataarray(
        f'/archive/Marc.Prange/discharge_statistics/'
        f'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/'
        f'flow_{days_per_year}_year-1_threshold_c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_'
        f'{start_year}_{end_year}.nc') 
        for days_per_year in days_per_year_list]
    store_low_high_flow_statistics(
        flow_threshold_list, threshold_kind_list, prctl_list, exp_name, 
        base_path, gfdl_processor, f'/archive/m2p/discharge_statistics/{exp_name}', 
        start_year, end_year, variable='rv_o_h2o', ar_condition=ar_condition
    )
    for threshold_kind, days_per_year in zip(threshold_kind_list, days_per_year_list):
        store_mean_low_high_flow_statistics(
            threshold_kind, exp_name, in_path, out_path, start_year, end_year,
                variable=f'{"ar_" if ar_condition else ""}flow_{days_per_year}_year-1'
            )


if __name__ == '__main__':
    main()
