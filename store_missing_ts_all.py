import xarray as xr
import numpy as np
import argparse
import data_util as du

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

def store_monthly_albedo(exp_name, out_path):
    albedo_dir = xr.open_mfdataset(
        f'{exp_basepath_map[exp_name]}{exp_name}/'
        'gfdl.ncrc5-intel23-classic-prod-openmp/pp/land/ts/monthly/1yr/'
        'land.*01-*12.albedo_dir.nc')
    start_year = str(albedo_dir.time[0].values)[:4]
    end_year = str(albedo_dir.time[-1].values)[:4]
    albedo_dif = xr.open_mfdataset(
        f'{exp_basepath_map[exp_name]}{exp_name}/'
        'gfdl.ncrc5-intel23-classic-prod-openmp/pp/land/ts/monthly/1yr/'
        'land.*01-*12.albedo_dif.nc')
    start_year = str(albedo_dif.time[0].values)[:4]
    end_year = str(albedo_dif.time[-1].values)[:4]
    albedo_dir.to_netcdf(
        f'{out_path}{exp_name}/ts_all/land.{start_year}01-{end_year}12.albedo_dir.nc')
    albedo_dif.to_netcdf(
        f'{out_path}{exp_name}/ts_all/land.{start_year}01-{end_year}12.albedo_dif.nc')

def store_monthly_grnd_flux(exp_name, out_path):
    grnd_flux = xr.open_mfdataset(
        f'{exp_basepath_map[exp_name]}{exp_name}/'
        'gfdl.ncrc5-intel23-classic-prod-openmp/pp/land/ts/monthly/1yr/'
        'land.*01-*12.grnd_flux.nc')
    start_year = str(grnd_flux.time[0].values)[:4]
    end_year = str(grnd_flux.time[-1].values)[:4]
    grnd_flux.to_netcdf(
        f'{out_path}{exp_name}/ts_all/land.{start_year}01-{end_year}12.grnd_flux.nc')

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

def store_monthly_precip_intensity_frequency(
        exp_name, out_path, vars=['pr', 'prsn'], subset='atmos', ar_day_mean=False):
    for var in vars:
        if exp_name == 'c192_obs':
            if var == 'ar_precip':
                ar_analysis = True
                load_var = 'precip'
            elif var == 'precip':
                ar_analysis = False 
                load_var = 'precip'
            pr = du.load_obs_data(
                'archive/Ming.Zhao/awg/2022.03/', year='*', 
                era5_variables=None, exp_name=exp_name, 
                ar_analysis=ar_analysis, obs_pr_dataset=subset,
                min_ar_pr_threshold=1/86400,
                low_pr_value=0,
                min_pr_var='precip',
                ar_masked_vars=['precip'],
                lon_180=False).sortby(
                    'time')
        else:
            if var == 'ar_precip':
                ar_analysis = True
                load_var = 'precip'
                if ar_day_mean:
                    low_pr_value = np.nan
                else:
                    low_pr_value = 0
            else:
                ar_analysis = False
            pr = du.load_model_data(
                    exp_basepath_map[exp_name], year='*', variables=[load_var],
                    exp_name=exp_name,
                    ar_analysis=ar_analysis,
                    min_ar_pr_threshold=1/86400,
                    low_pr_value=low_pr_value,
                    min_pr_var='precip',
                    ar_masked_vars=[load_var],
                    gfdl_processor='gfdl*',
                    lon_180=False)
        start_year = str(pr.time[0].values)[:4]
        end_year = str(pr.time[-1].values)[:4]
        pr_intensity = pr[var].where(pr[var]*86400 > 1)
        pr_intensity_monthly = pr_intensity.resample(time="1MS").mean(dim="time").rename(f'{var}_intensity')
        pr_frequency_monthly = pr_intensity.resample(time="1MS").count(dim="time").rename(f'{var}_frequency')
        pr_intensity_monthly.to_netcdf(
                f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.{var}_intensity.nc')
        pr_frequency_monthly.to_netcdf(
                f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.{var}_frequency.nc')
        if var == 'ar_precip':
            if ar_day_mean:
                rename_var = 'ar_day_precip'
            else:
                rename_var = 'ar_precip'
            ar_pr_monthly = pr[var].resample(time="1MS").mean(dim="time").rename(rename_var)
            ar_pr_monthly.to_netcdf(
                f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.{rename_var}.nc')
        if exp_name == 'c192_obs':
            if var == 'ar_precip':
                pr_mean_monthly = pr[var].resample(time="1MS").mean(dim="time").rename('ar_precip')
                pr_mean_monthly.to_netcdf(
                    f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.ar_precip.nc')
            else:
                pr_mean_monthly = pr[var].resample(time="1MS").mean(dim="time").rename('precip')
                pr_mean_monthly.to_netcdf(
                    f'{out_path}{exp_name}/ts_all/{subset}.{start_year}01-{end_year}12.precip.nc')

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K')
    args = parser.parse_args()
    exp_name = args.exp_name
    # store_monthly_plevel_winds(
    #     exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min',
    #     plevels=[700, 250],
    #     out_path='/archive/Marc.Prange/ts_all_missing_vars/'
    # )
    # store_monthly_precip_intensity_frequency(
    #     exp_name=exp_name,
    #     out_path='/archive/Marc.Prange/ts_all_missing_vars/',
    #     vars=['ar_precip', 'precip'],
    #     subset='imerg'
    #     )
    store_monthly_grnd_flux(
        exp_name=exp_name,
        out_path='/archive/Marc.Prange/ts_all_missing_vars/'
    )


if __name__ == '__main__':
    _main()
