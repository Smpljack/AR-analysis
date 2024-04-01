import xarray as xr
import numpy as np
from pathlib import Path
import multiprocessing
import argparse


"""
Module Name: precipitation_utils

This module provides utility functions for processing and analyzing precipitation data using xarray and multiprocessing.

Functions:
- lon_360_to_180(ds): Converts longitudes from 0-360 to -180 to 180.
- sel_na_pacific(ds): Selects data within the North Pacific region.
- sel_na_westcoast(ds): Selects data within the North American West Coast region.
- sel_california(ds): Selects data within the California region.
- sel_na(ds): Selects data within the broader North America region.
- load_ar_data(base_path, exp_name, year): Loads Atmospheric River (AR) data.
- load_pr_obs_data(base_path, exp_name, year): Loads precipitation observation data.
- load_model_data(base_path, year, variables, exp_name, ar_analysis, gfdl_processor): Loads model data.
- load_era5_data(base_path, year, variables, ar_analysis, mswep_precip, exp_name): Loads ERA5 data.
- load_daily_sharc_data(base_path, exp_name, year): Loads daily SHARC data.
- daily_ar_shape(ar_data, days): Computes daily AR shape data.
- ar_rain_rate(data): Computes AR rain rate.
- store_loc_model_data(start_year, end_year, loc_lat, loc_lon, loc_name, model_data, base_path, exp_name, out_base_path, variables): Stores model data for a specific location.
- store_yearly_NA_model_data(year, base_path, exp_name, out_base_path, gfdl_processor, variables, ar_analysis): Stores yearly North America model data.
- store_loc_obs_data(start_year, end_year, loc_lon, loc_lat, loc_name, base_path, exp_name, variables): Stores observation data for a specific location.
- load_loc_data(start_year, end_year, exp_name, loc_name, base_path): Loads location-specific data.
- get_strong_precip_days(data, min_precip, ar_day, precip_var): Gets strong precipitation days.
- create_precip_event_mask(data, min_precip, ar_day, precip_var, min_days_between_events, days_ahead, days_back): Creates a precipitation event mask.
- create_temporal_composite_ds(data, days, days_back, days_ahead, min_days_between_events): Creates a temporal composite dataset.
- get_independent_event_days(days, min_days_between_events): Gets independent event days.
- store_loc_composite_ds(start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_day): Stores location composite dataset.
- load_loc_composite_ds(start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_str): Loads location composite dataset.
- load_sharc_composite_ds(start_year, end_year, base_path, exp_name, min_precip, ar_str): Loads SHARC composite dataset.
- store_na_composite_mean_ds(data, exp_name, out_base_path, min_precip, ar_day, queue, lon_min, lon_max): Stores North America composite mean dataset.
- temporal_composite_ds_grouped_for_precip(data_loc, min_precip, ar_day): Creates grouped temporal composite dataset for precipitation.
- temporal_composite_ds_mean_for_precip(data_loc, min_precip, ar_day): Creates mean temporal composite dataset for precipitation.
- store_na_temporal_comp_for_lon_range(na_data, lon_min, lon_max, exp_name, out_basepath, min_precip, ar_day): Stores North America temporal composite for a given longitude range.
- _main(): Main function to execute tasks such as loading model data and storing datasets.

Note: Some functions may require additional dependencies not mentioned in the docstrings.
"""

def lon_360_to_180(ds):
    """
    Converts longitudes from 0-360 to -180 to 180.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray): Input dataset with longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray: Dataset or DataArray with converted longitudes.
    """
    ds['lon'] = xr.where(ds.lon > 180, ds.lon - 360, ds.lon)
    ds = ds.sortby('lon')
    return ds

def sel_na_pacific(ds):
    """
    Selects data within the North Pacific region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray): Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray: Dataset or DataArray containing data within the North Pacific region.
    """
    return ds.sel({'lat': slice(10, 60), 'lon': slice(-170, -80)})

def sel_na_westcoast(ds):
    """
    Selects data within the North American West Coast region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray): Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray: Dataset or DataArray containing data within the North American West Coast region.
    """
    return ds.sel(
            {
            'lat': slice(20, 65),
            'lon': slice(-135, -105)
            })          

def sel_california(ds):
    """
    Selects data within the California region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray): Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray: Dataset or DataArray containing data within the California region.
    """
    return ds.sel({'lat': slice(31, 44), 'lon': slice(-130, -110)})

def sel_na(ds):
    """
    Selects data within the broader North America region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray): Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray: Dataset or DataArray containing data within the broader North America region.
    """
    return ds.sel({'lat': slice(20, 70), 'lon': slice(-140, -60)})

def load_ar_data(base_path, exp_name, year):
    """
    Loads Atmospheric River (AR) data.

    Parameters:
    - base_path (str): Base path for the data files.
    - exp_name (str): Experiment name.
    - year (int): Year of the data.

    Returns:
    - xarray.Dataset: AR data for the specified year.
    """
    ar_data = xr.open_dataset(
            base_path+f'{exp_name}/AR_climlmt/{exp_name}_AR_{year}.nc')
    return ar_data

def load_pr_obs_data(base_path, exp_name, year):
    """
    Loads observed precipitation data based on MSWEP.

    Parameters:
    - base_path (str): Base path for the data files.
    - exp_name (str): Experiment name.
    - year (int): Year of the data.

    Returns:
    - xarray.Dataset: Observed precipitation data for the specified year.
    """

    pr_data = xr.open_dataset(
        base_path+f'{exp_name}/atmos_data/daily_mswep/mswep.{year}0101-{year}1231.precipitation.nc')
    return pr_data

def load_model_data(
    base_path, year, variables, 
    exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020', 
    ar_analysis=True, 
    gfdl_processor='gfdl.ncrc4-intel-prod-openmp'):
    """
    Loads model data.

    Parameters:
    - base_path (str): Base path for the data files.
    - year (int): Year of the data.
    - variables (list): List of variables to load.
    - exp_name (str): Experiment name. Default is 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020'.
    - ar_analysis (bool): Whether to perform Atmospheric River (AR) analysis. Default is True.
    - gfdl_processor (str): Processor name. Default is 'gfdl.ncrc4-intel-prod-openmp'.

    Returns:
    - xarray.Dataset: Model data for the specified year and variables.
    """
    print(f'Loading model data for year {year}\nexp_name: {exp_name}', flush=True)
    cmip_land_vars = ['snw', 'snc', 'mrro', 'mrso', 'mrsos', 'snm', 'tran']
    land_vars = ['evap_land', 'npp', 'precip', 'runf_soil', 'runf_lake', 
                 'soil_ice', 'soil_liq', 't_ref_max', 't_ref_min', 
                 'vegn_temp_max', 'vegn_temp_min']
    river_variables = ['rv_d_h2o', 'rv_o_h2o']
    data = xr.merge(
        [xr.open_dataset(
             base_path+f'{exp_name}/{gfdl_processor}/pp/atmos_cmip/ts/daily/1yr/'
             f'atmos_cmip.{year}0101-{year}1231.{var}.nc')
             for var in variables 
             if var not in ['ivtx', 'ivty'] and var not in land_vars and var not in cmip_land_vars and var not in river_variables])
    if np.isin(variables, cmip_land_vars).any():
        data = xr.merge(
            [data,
             xr.merge(
                [xr.open_dataset(
                    base_path+f'{exp_name}/{gfdl_processor}/pp/land_cmip/ts/daily/1yr/'
                    f'land_cmip.{year}0101-{year}1231.{var}.nc')
                 for var in variables if var in cmip_land_vars])]
        )
    if np.isin(variables, land_vars).any():
        data = xr.merge(
            [data, 
             xr.merge(
                [xr.open_dataset(
                    base_path+f'{exp_name}/{gfdl_processor}/pp/land/ts/daily/1yr/'
                    f'land.{year}0101-{year}1231.{var}.nc')
                 for var in variables if var in land_vars])]
        )
    data['prli'] = data.pr - data.prsn
    data = lon_360_to_180(data)
    data['time'] = np.array(data.time.values, dtype='datetime64[1D]')
    if 'ivtx' in variables:
        ivt_data = xr.merge(
            [xr.open_dataset(base_path+f'{exp_name}/{gfdl_processor}/pp/atmos_cmip/ts/6hr/1yr/'
            f'atmos_cmip.{year}010100-{year}123123.{var}.nc')
             for var in ['ivtx', 'ivty']]).resample({'time': '1D'}).mean()
        ivt_data = lon_360_to_180(ivt_data)
        ivt_data = ivt_data.assign_coords({'lat': data.lat, 'lon': data.lon})
        ivt_data['time'] = np.array(ivt_data.time.values, dtype='datetime64[1D]') 
        ivt_data = ivt_data.isel(time=np.isin(ivt_data.time, data.time))
        data = xr.merge([data, ivt_data], compat='override')
    if np.isin(river_variables, variables).any():
        river_data = xr.merge(
            [xr.open_dataset(base_path+f'{exp_name}/{gfdl_processor}/pp/river/ts/daily/1yr/'
            f'river.{year}0101-{year}1231.{var}.nc')
             for var in ['rv_o_h2o', 'rv_d_h2o']])
        river_data = lon_360_to_180(river_data)
        river_data['time'] = np.array(river_data.time.values, dtype='datetime64[1D]')
        data = xr.merge([data, river_data], compat='override') 
    if ar_analysis:
        model_ar_data = load_ar_data(base_path, exp_name, year)
        model_ar_data = lon_360_to_180(model_ar_data)
        model_ar_data = model_ar_data.assign_coords({'lat': data.lat, 'lon': data.lon})
        data['ar_shape'] = daily_ar_shape(model_ar_data, data.time)
        data['ar_pr'] = ar_rain_rate(data, 'pr')
        data['ar_prsn'] = ar_rain_rate(data, 'prsn')
        data['ar_prli'] = ar_rain_rate(data, 'prli')
    return data


def load_era5_data(base_path, year, variables, ar_analysis=True, mswep_precip=True, exp_name='c192_obs'):
    """
    Loads ERA5 data.

    Parameters:
    - base_path (str): Base path for the data files.
    - year (int): Year of the data.
    - variables (list): List of variables to load.
    - ar_analysis (bool): Whether to perform Atmospheric River (AR) analysis. Default is True.
    - mswep_precip (bool): Whether to load MSWEP precipitation data. Default is True.
    - exp_name (str): Experiment name. Default is 'c192_obs'.

    Returns:
    - xarray.Dataset: ERA5 data for the specified year and variables.
    """

    data = xr.merge(
        [xr.open_dataset(
             base_path+
             f'{exp_name}/atmos_data/daily_era5/ERA5.{year}0101-{year}1231.{var}.nc') for var in variables])
    data = lon_360_to_180(data)
    data['time'] = np.array(data.time.values, dtype='datetime64[1D]') 
    if mswep_precip:
        obs_pr_data = load_pr_obs_data(base_path, exp_name, year)
        obs_pr_data = lon_360_to_180(obs_pr_data)
        data = xr.merge([data, obs_pr_data]).rename({'precipitation': 'pr'}) 
        data['pr'] = data.pr / 86400
    if ar_analysis:
        ar_data = load_ar_data(base_path, exp_name, year)
        ar_data = lon_360_to_180(ar_data)
        data['ar_shape'] = daily_ar_shape(ar_data, data.time)
        data['ar_pr'] = ar_rain_rate(data)
    return data

def load_daily_sharc_data(
    base_path='/archive/Marc.Prange/LM4p2_SHARC', 
    exp_name='providence_lm4sharc_ksat0016_angle087rad_ep26_114y', 
    year='all_years'):
    """
    Loads daily SHARC data.

    Parameters:
    - base_path (str): Base path for the data files. Default is '/archive/Marc.Prange/LM4p2_SHARC'.
    - exp_name (str): Experiment name. Default is 'providence_lm4sharc_ksat0016_angle087rad_ep26_114y'.
    - year (int or str): Year of the data. Default is 'all_years'.

    Returns:
    - xarray.Dataset: Daily SHARC data for the specified year and experiment.
    """

    print(f'Loading sharc atmos data for year {year}...', flush=True)
    sharc_atmos = xr.open_dataset(
        f'{base_path}/{exp_name}/history/19010101.atmos_month.nc')
    sharc_atmos['time'] = sharc_atmos.time.astype('datetime64[ns]')
    if year != 'all_years':
        sharc_atmos = sharc_atmos.sel(time=f'{year}')
    print(f'Loading sharc ptid data for year {year}...', flush=True)
    sharc_data = xr.open_dataset(
        f'{base_path}/{exp_name}/history/19010101.ptid_diurnal.nc')
    sharc_data['time'] = sharc_data.time.astype('datetime64[ns]')
    if year != 'all_years':   
        sharc_data = sharc_data.sel(time=f'{year}')
    sharc_data = sharc_data.mean('time_of_day_24')
    sharc_data = xr.merge([sharc_data.squeeze(), sharc_atmos.squeeze()], compat='override')
    return sharc_data

def daily_ar_shape(ar_data, days):
    """
    Maps 6hourly AR shape data to daily AR data.
    If any point in 6hourly is AR on given day, point is flagged as AR day.
    Parameters:
    - ar_data (xarray.Dataset): AR data.
    - days (array-like): Array-like object containing days.

    Returns:
    - xarray.DataArray: Daily AR shape data.
    """

    daily_ar_shape = xr.DataArray(
        coords={
            'time': days,
            'lat': ar_data.lat,
            'lon': ar_data.lon,
        },
        data=np.ones((len(days), len(ar_data.lat), len(ar_data.lon))) * np.nan
    )
    for iday, day in enumerate(days):
        daily_ar_shape[iday, :, :] = xr.where(
            ar_data.sel(time=slice(day, day+np.timedelta64(23, 'h'))).shape.sum('time') > 0, True, np.nan)
    return daily_ar_shape

def ar_rain_rate(data, pr_var='pr'):
    return data[f'{pr_var}'].where(data.ar_shape == 1, other=0)

def store_loc_model_data(
        start_year, end_year, 
        loc_lat, loc_lon,
        loc_name,
        model_data=None,
        base_path='/archive/Ming.Zhao/awg/2022.03/', 
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
        out_base_path='/archive/Marc.Prange/',
        variables=['ts', 'prw', 'pr', 'prsn', 'ivtx', 'ivty', 'mrro', 'mrsos', 'mrso', 'snw', 'evap_land', 'precip', 'rv_d_h2o', 'rv_o_h2o']):
    if model_data is None:
        model_data_loaded = False
    else:
        model_data_loaded = True
            
    for year in np.arange(start_year, end_year+1):
        if not model_data_loaded:
            model_data = load_model_data(
                base_path, year, 
                exp_name=exp_name, 
                variables=variables, ar_analysis=True)
        model_data_loc = model_data.isel(
            time=model_data['time.year']==year).sel(
                {'lon': loc_lon, 'lat': loc_lat}, method='nearest')
        print(f'Storing {loc_name} data for {year}...', flush=True)
        dir = f'{out_base_path}/{loc_name}_data/{exp_name}/'
        Path(dir).mkdir(parents=True, exist_ok=True)
        model_data_loc.to_netcdf(
            f'{dir}{exp_name}_{loc_name}_{year}.nc')

def store_yearly_NA_model_data(
        year,
        base_path='/archive/Ming.Zhao/awg/2022.03/', 
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
        out_base_path='/archive/Marc.Prange/',
        gfdl_processor='gfdl.ncrc4-intel-prod-openmp',
        variables=['ts', 'prw', 'pr', 'prsn', 'ivtx', 'ivty', 'mrro', 'mrsos', 'mrso', 'snw', 'evap_land', 'precip'],
        ar_analysis=True):
    print(f'Loading model data for {year}...', flush=True)
    model_data = load_model_data(
        base_path, year, 
        exp_name=exp_name, 
        gfdl_processor=gfdl_processor,
        variables=variables, ar_analysis=ar_analysis)
    model_data_na = sel_na(model_data)
    print(f'Storing NA data for {year}...', flush=True)
    dir = f'{out_base_path}/na_data/{exp_name}/'
    Path(dir).mkdir(parents=True, exist_ok=True)
    model_data_na.to_netcdf(
        f'{dir}{exp_name}_na_{year}.nc')

def store_loc_obs_data(
        start_year, end_year, 
        loc_lon, loc_lat,
        loc_name,
        base_path='/archive/Ming.Zhao/awg/2022.03/', 
        exp_name='c192_obs',
        variables=['ivtx', 'ivty', 'prw']):
    for year in np.arange(start_year, end_year+1):
        print(f'Loading obs data for {year}...', flush=True)
        era5_data = load_era5_data(base_path, year, variables, ar_analysis=True)    
        print(f'Storing {loc_name} obs data for {year}...', flush=True)
        era5_data_loc = era5_data.sel({'lon': loc_lon, 'lat': loc_lat})
        era5_data_loc.to_netcdf(
            f'/archive/Marc.Prange/{loc_name}_data/{exp_name}/{exp_name}_{loc_name}_{year}.nc')
        
def load_loc_data(start_year, end_year, exp_name, loc_name, base_path='/archive/Marc.Prange/'):
    print(f'Loading loc_data for {loc_name}', flush=True)
    return xr.concat([xr.open_dataset(
        f'{base_path}{loc_name}_data/'
        f'{exp_name}/{exp_name}_{loc_name}_{year}.nc')
        for year in range(start_year, end_year+1)], dim='time')

def get_strong_precip_days(data, min_precip=30, ar_day=True, precip_var='pr', winter=False):
    filter = (data[f'{precip_var}'] > min_precip/86400)
    if ar_day:
        filter &= (data.ar_shape == 1)
    if winter:
        print("Filtering for winter events...", flush=True)
        data = data.isel(time=np.isin(data['time.month'], [11, 12, 1, 2, 3]))
    strong_precip_days = data.time.where(filter).dropna('time')
    return strong_precip_days

def create_precip_event_mask(
        data, min_precip, ar_day=False, precip_var='pr', 
        min_days_between_events=3, days_ahead=5, days_back=5):
    strong_precip_mask = xr.DataArray(
        data=data[f'{precip_var}']*False, 
        name='strong_precip_mask')
    for ilat, lat in enumerate(data.lat.values):
        print('Creating mask of strong precip days for '
              f'(lat:{np.round(lat, 2)}/{np.round(data.lat.max().values)})', 
              flush=True)
        for ilon, lon in enumerate(data.lon.values):
            precip_days = get_strong_precip_days(
                data.sel({'lat': lat, 'lon': lon}), min_precip, ar_day, precip_var=precip_var)
            # Filter days on edge of timeseries
            precip_days = precip_days.where(
                (np.datetime64(data.time.values[-1]) - precip_days) > 
                 np.timedelta64(days_ahead, 'D'), drop=True)
            precip_days = precip_days.where(
                (precip_days - np.datetime64(data.time.values[0])) > 
                 np.timedelta64(days_back, 'D'), drop=True)
            # Make sure events are independent
            precip_days = get_independent_event_days(precip_days.values, min_days_between_events)
            strong_precip_mask[:, ilat, ilon] = data.time.isin(precip_days)
    return strong_precip_mask

def create_temporal_composite_ds(data, days, days_back=5, days_ahead=5, min_days_between_events=3):
    print("Creating temporal composite for loc "
         f"{np.round(data.lat.values, 2)}, {np.round(data.lon.values, 2)}", 
         flush=True)
    days = days.where((np.datetime64(data.time.values[-1]) - days) > np.timedelta64(days_ahead, 'D'), drop=True)
    days = days.where((days - np.datetime64(data.time.values[0])) > np.timedelta64(days_back, 'D'), drop=True)
    days = get_independent_event_days(days.values, min_days_between_events)
    if len(days) == 0:
        comp_ds = xr.Dataset(
            coords={
                'lat': data.lat, 
                'lon': data.lon, 
                'case': [], 
                'time': np.arange(-days_back, days_ahead+1, dtype='timedelta64[D]').astype('timedelta64[ns]')
            })
    else:
        comp_ds = xr.concat([data.sel(
            time=slice(day-np.timedelta64(days_back, 'D'), 
                    day+np.timedelta64(days_ahead, 'D'))
                    ).assign_coords(
                        {
                            'time': np.arange(-days_back, days_ahead+1, dtype='timedelta64[D]').astype('timedelta64[ns]'),
                            'case': case,
                        }
                    ).assign(
                        variables=
                        {
                            'strong_precip_date': 
                                (('time'), np.arange(
                                    day-np.timedelta64(days_back, 'D'), 
                                    day+np.timedelta64(days_ahead+1, 'D'), 
                                    dtype='datetime64[D]').astype('datetime64[ns]')),
                        }
                    ) for case, day in enumerate(days)], dim='case')
        print(f"Found {len(comp_ds.case)} precip events.", flush=True)
    return comp_ds

def get_independent_event_days(days, min_days_between_events=3):
    for time in days:
        if time in days:
            close_dates = np.arange(time, time + np.timedelta64(min_days_between_events+1, 'D'), dtype='datetime64[D]')[1:]
            days = days[np.isin(days, close_dates, invert=True)]
    return days

def store_loc_composite_ds(start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_day):
    comp_data = create_loc_composite_ds(
        start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_day)
    if ar_day:
        ar_str = 'AR_days'
    else:
        ar_str = 'all_days'
    dir = f'{base_path}/{loc_name}_data/{exp_name}/precip_composite/'
    Path(dir).mkdir(parents=True, exist_ok=True)
    comp_data.to_netcdf(
        dir+f'{exp_name}_{loc_name}_{ar_str}_min_precip_{min_precip}_{start_year}-{end_year}_temporal_composite.nc')

def store_sharc_composite_ds(start_year, end_year, base_path, exp_name, min_precip, ar_day, precip_var='pr', winter=False):
    data = xr.concat(
        [load_daily_sharc_data(base_path, exp_name, year) for year in range(start_year, end_year+1)],
        dim='time')
    data['pr'] = data.lprec + data.fprec
    precip_days = get_strong_precip_days(data, min_precip, ar_day, precip_var, winter) 
    comp_data = create_temporal_composite_ds(data, precip_days)
    dir = f'{base_path}/{exp_name}/precip_composite/'
    Path(dir).mkdir(parents=True, exist_ok=True)
    if ar_day:
        ar_str = 'AR_days'
    else:
        ar_str = 'all_days'
    comp_data.to_netcdf(
        dir+f'{exp_name}_{ar_str}_min_{precip_var}_{min_precip}_{start_year}-{end_year}_temporal_composite.nc') 

def create_loc_composite_ds(start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_day):
    data = load_loc_data(start_year, end_year, exp_name, loc_name, base_path)
    precip_days = get_strong_precip_days(data, min_precip, ar_day)
    comp_data = create_temporal_composite_ds(data, precip_days)
    return comp_data

def load_loc_composite_ds(start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_str):
    return xr.open_dataset(
        f'{base_path}/{loc_name}_data/{exp_name}/precip_composite/'
        f'{exp_name}_{loc_name}_{ar_str}_min_precip_{min_precip}_{start_year}-{end_year}_temporal_composite.nc'
    )

def load_sharc_composite_ds(start_year, end_year, base_path, exp_name, min_precip, ar_str, precip_var='pr'):
    return xr.open_dataset(
        f'{base_path}/{exp_name}/precip_composite/'
        f'{exp_name}_{ar_str}_min_{precip_var}_{min_precip}_{start_year}-{end_year}_temporal_composite.nc')  

def store_na_composite_mean_ds(data, exp_name, out_base_path, min_precip, precip_var, ar_day, winter, queue=None, lon_min=None, lon_max=None):
    if lon_min is not None:
        data = data.sel(lon=slice(lon_min, lon_max))
    comp_data = xr.concat([
        xr.concat([
            temporal_composite_ds_mean_for_precip(
                data.sel({'lat': lat, 'lon': lon}),
                min_precip=min_precip, 
                precip_var=precip_var,
                ar_day=ar_day, 
                winter=winter)
                for lat in data.lat], dim='lat', coords='minimal')
                for lon in data.lon], dim='lon', coords='minimal')
    if ar_day:
        ar_str = 'AR_days'
    else:
        ar_str = 'all_days'
    if queue is not None:
        queue.put(comp_data)
    else:
        comp_data.to_netcdf(
            f'{out_base_path}'
            f'{exp_name}_na_{ar_str}_min_{precip_var}_{min_precip}_1990-2020_temporal_composite_mean.nc',)
            # encoding={'mrro_nfpr_frac_bins': {'dtype': 'str'}})

def temporal_composite_ds_grouped_for_precip(data_loc, min_precip, ar_day):
    precip_days = get_strong_precip_days(data_loc, min_precip, ar_day)
    data_loc['mrro_nfpr_frac'] = data_loc.mrro / (data_loc.pr - data_loc.prsn)
    mrro_nfpr_frac_bins = np.arange(0, 1.5, 0.25)
    comp_data_loc = create_temporal_composite_ds(data_loc, precip_days) 
    comp_data_loc_grouped = comp_data_loc.groupby_bins(
        comp_data_loc.mrro_nfpr_frac.isel(time=9), 
        mrro_nfpr_frac_bins, 
        include_lowest=True).mean()
    return comp_data_loc_grouped

def temporal_composite_ds_mean_for_precip(data_loc, min_precip=20, precip_var='pr', ar_day=False, winter=False):
    precip_days = get_strong_precip_days(data_loc, min_precip, ar_day, precip_var=precip_var, winter=winter)
    data_loc['mrro_nfpr_frac'] = data_loc.mrro / (data_loc.pr - data_loc.prsn)
    comp_data_loc = create_temporal_composite_ds(data_loc, precip_days)
    comp_data_loc_mean = comp_data_loc.mean('case')
    comp_data_loc_mean['strong_precip_event_count'] = len(comp_data_loc.case)
    return comp_data_loc_mean

def store_na_temporal_comp_for_lon_range(
        na_data, lon_min, lon_max, 
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
        out_basepath='/archive/Marc.Prange/',
        min_precip=20,
        precip_var='pr',
        ar_day=False,
        winter=False):
    na_data = na_data.sel(lon=slice(lon_min, lon_max))
    q = multiprocessing.Queue()
    process_lon_bounds = np.linspace(na_data.lon.min(), na_data.lon.max(), 9)

    processes = []
    for plon_min, plon_max in zip(process_lon_bounds[:-1], process_lon_bounds[1:]):
        print(f'Started process for lon interval: {plon_min}-{plon_max}', flush=True)
        p = multiprocessing.Process(
            target=store_na_composite_mean_ds, 
            args=(na_data, exp_name, out_basepath, min_precip, precip_var, ar_day, winter),
            kwargs={'queue': q, 'lon_min': plon_min, 'lon_max': plon_max})
        processes.append(p)
        p.start()

    print('Concat.', flush=True)
    composite_mean_list = [q.get() for p in processes]
    for p in processes:
        print('Joining process.', flush=True)
        p.join()
    print('Concatenating data...', flush=True)
    composite_mean_ds = xr.concat(composite_mean_list, dim='lon').sortby('lon')
    print('Storing data...', flush=True)
    outpath = f'{out_basepath}na_data/{exp_name}/temporal_composite/'
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if ar_day:
        ar_str = 'AR_days'
    else:
        ar_str = 'all_days'
    if winter:
        winter_str = 'Nov-Mar'
    else:
        winter_str = 'all_months'
    composite_mean_ds.to_netcdf(
            outpath +
            f'{exp_name}_na_{ar_str}_min_{precip_var}_{min_precip}_1990-2020_temporal_composite_mean_{winter_str}_lon_'
            f'{np.round(lon_min, 2)}-{np.round(lon_max, 2)}.nc',)

def _main():
    # na_data = xr.open_mfdataset(
    #     '/archive/Marc.Prange/na_data/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
    #     'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020_na_*.nc').load()
    # store_loc_model_data(
    #     start_year=2020, end_year=2021, 
    #     loc_lat=45.515, loc_lon=-122.678,
    #     loc_name='portland',
    #     model_data=na_data,
    #     base_path='/archive/Marc.Prange/', 
    #     exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
    #     out_base_path='/archive/Marc.Prange/',)
    # store_loc_composite_ds(
    #     start_year=1990, end_year=2019, 
    #     loc_name='portland', 
    #     base_path='/archive/Marc.Prange/', 
    #     exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020', 
    #     min_precip=20, 
    #     ar_day=False)
    # na_data = xr.open_mfdataset(
    #     '/archive/Marc.Prange/na_data/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
    #     'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020_na_*.nc').load()
    # # data_loc = na_data.isel({'lon': 10, 'lat': 10}) 
    # store_na_composite_mean_ds(
    #     na_data, 
    #     'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020', 
    #     out_base_path='/archive/Marc.Prange', 
    #     min_precip=20, 
    #     precip_var='prli',
    #     ar_day=False,
    #     winter=True,
    #     lon_min=-130,
    #     lon_max=-129)
    # temporal_composite_ds_mean_for_precip(data_loc, min_precip=20, ar_day=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=1990)
    args = parser.parse_args()
    store_yearly_NA_model_data(
        args.year,
        base_path='/archive/Ming.Zhao/awg/2023.04/', 
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K',
        out_base_path='/archive/Marc.Prange/',
        gfdl_processor='gfdl.ncrc5-intel23-classic-prod-openmp',
        variables=['ts', 'prw', 'pr', 'prsn', 'ivtx', 'ivty', 'mrro', 'mrsos', 'mrso', 'snw', 'evap_land', 'precip', 'rv_d_h2o', 'rv_o_h2o'],
        ar_analysis=True)

if __name__ == '__main__':
    _main()