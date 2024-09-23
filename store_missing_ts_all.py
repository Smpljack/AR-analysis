import xarray as xr
import numpy as np
import argparse


exp_basepath_map = {
        'c192L33_am4p0_amip_HIRESMIP_HX': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_HX_p2K': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K': '/archive/Ming.Zhao/awg/2023.04/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min': '/archive/Ming.Zhao/awg/2023.04/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K': '/archive/Ming.Zhao/awg/2023.04/',
        'c192_obs': '/archive/Ming.Zhao/awg/2022.03/'
    }

def store_monthly_plevel_winds(exp_name, plevels, out_path):
    u = xr.open_mfdataset(
        f'{exp_basepath_map[exp_name]}{exp_name}/ts_all/atmos.*.ucomp.nc')
    v = xr.open_mfdataset(
        f'{exp_basepath_map[exp_name]}{exp_name}/ts_all/atmos.*.vcomp.nc')
    start_year = str(u.time[0].values)[:4]
    end_year = str(u.time[-1].values)[:4]
    for plevel in plevels:
        ulevel = u.sel(level=plevel).rename({'ucomp': f'u_{plevel}'})
        vlevel = v.sel(level=plevel).rename({'vcomp': f'v_{plevel}'})
        ulevel.to_netcdf(
            f'{out_path}{exp_name}/ts_all/atmos.{start_year}01-{end_year}12.u_{plevel}.nc')
        vlevel.to_netcdf(
            f'{out_path}{exp_name}/ts_all/atmos.{start_year}01-{end_year}12.v_{plevel}.nc')

def store_monthly_precip_intensity_frequency(exp_name, out_path, vars=['pr', 'prsn']):
    for var in vars:
        if exp_name == 'c192_obs':
            pr = xr.open_mfdataset(
                f'{exp_basepath_map[exp_name]}{exp_name}/atmos_data/daily_mswep/mswep.*.{var}.nc')
            pr[var] = pr[var]/86400 #MSWEP given in mm/day
            subset = 'mswep'
        else:
            pr = xr.open_mfdataset(
                f'{exp_basepath_map[exp_name]}{exp_name}/gfdl*/pp/atmos_cmip/ts/daily/1yr/atmos_cmip.*.{var}.nc')
            subset = 'atmos_cmip'
        start_year = str(pr.time[0].values)[:4]
        end_year = str(pr.time[-1].values)[:4]
        pr_intensity = pr[var].where(pr[var]*86400 > 1)
        pr_intensity_monthly = pr_intensity.resample(time="1MS").mean(dim="time").rename('pr_intensity')
        pr_frequency_monthly = pr_intensity.resample(time="1MS").count(dim="time").rename('pr_frequency')
        pr_intensity_monthly.to_netcdf(
                f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.pr_intensity.nc')
        pr_frequency_monthly.to_netcdf(
                f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.pr_frequency.nc')
        if exp_name == 'c192_obs':
            pr_mean_monthly = pr[var].resample(time="1MS").mean(dim="time").rename('precip')
            pr_mean_monthly.to_netcdf(
                f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.precip.nc')


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='c192_obs')
    args = parser.parse_args() 
    exp_name = args.exp_name
    # store_monthly_plevel_winds(
    #     exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min',
    #     plevels=[700, 250],
    #     out_path='/archive/Marc.Prange/ts_all_missing_vars/'
    # )
    store_monthly_precip_intensity_frequency(
        exp_name=exp_name,
        out_path='/archive/Marc.Prange/ts_all_missing_vars/',
        vars=['precipitation']
        )


if __name__ == '__main__':
    _main()