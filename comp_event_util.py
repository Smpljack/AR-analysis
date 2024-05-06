import xarray as xr
import numpy as np
from pathlib import Path
import multiprocessing

import data_util as du


def store_loc_model_data(
        start_year, end_year, 
        loc_lat, loc_lon,
        loc_name,
        model_data=None,
        base_path='/archive/Ming.Zhao/awg/2022.03/', 
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
        out_base_path='/archive/Marc.Prange/',
        variables=['ts', 'prw', 'pr', 'prsn', 'ivtx', 'ivty', 'mrro', 'mrsos', 
                   'mrso', 'snw', 'evap_land', 'precip', 'rv_d_h2o', 'rv_o_h2o']):
    if model_data is None:
        model_data_loaded = False
    else:
        model_data_loaded = True
            
    for year in np.arange(start_year, end_year+1):
        if not model_data_loaded:
            model_data = du.load_model_data(
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


def store_loc_obs_data(
        start_year, end_year, 
        loc_lon, loc_lat,
        loc_name,
        base_path='/archive/Ming.Zhao/awg/2022.03/', 
        exp_name='c192_obs',
        variables=['ivtx', 'ivty', 'prw']):
    for year in np.arange(start_year, end_year+1):
        print(f'Loading obs data for {year}...', flush=True)
        era5_data = du.load_era5_data(base_path, year, variables, ar_analysis=True)    
        print(f'Storing {loc_name} obs data for {year}...', flush=True)
        era5_data_loc = era5_data.sel({'lon': loc_lon, 'lat': loc_lat})
        era5_data_loc.to_netcdf(
            f'/archive/Marc.Prange/{loc_name}_data/{exp_name}/'
             '{exp_name}_{loc_name}_{year}.nc')
        
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
            precip_days = du.get_strong_precip_days(
                data.sel({'lat': lat, 'lon': lon}), 
                min_precip, ar_day, precip_var=precip_var)
            # Filter days on edge of timeseries
            precip_days = precip_days.where(
                (np.datetime64(data.time.values[-1]) - precip_days) > 
                 np.timedelta64(days_ahead, 'D'), drop=True)
            precip_days = precip_days.where(
                (precip_days - np.datetime64(data.time.values[0])) > 
                 np.timedelta64(days_back, 'D'), drop=True)
            # Make sure events are independent
            precip_days = get_independent_event_days(
                precip_days.values, min_days_between_events)
            strong_precip_mask[:, ilat, ilon] = data.time.isin(precip_days)
    return strong_precip_mask

def create_temporal_composite_ds(
    data, days, days_back=5, days_ahead=5, min_days_between_events=3):
    print("Creating temporal composite for loc "
         f"{np.round(data.lat.values, 2)}, {np.round(data.lon.values, 2)}", 
         flush=True)
    days = days.where(
        (np.datetime64(data.time.values[-1]) - days) > 
        np.timedelta64(days_ahead, 'D'), drop=True)
    days = days.where(
        (days - np.datetime64(data.time.values[0])) > 
        np.timedelta64(days_back, 'D'), drop=True)
    days = get_independent_event_days(days.values, min_days_between_events)
    if len(days) == 0:
        comp_ds = xr.Dataset(
            coords={
                'lat': data.lat, 
                'lon': data.lon, 
                'case': [], 
                'time': np.arange(
                    -days_back, days_ahead+1, 
                    dtype='timedelta64[D]').astype('timedelta64[ns]')
            })
    else:
        comp_ds = xr.concat([data.sel(
            time=slice(day-np.timedelta64(days_back, 'D'), 
                    day+np.timedelta64(days_ahead, 'D'))
                    ).assign_coords(
                        {
                            'time': np.arange(
                                -days_back, days_ahead+1, 
                                dtype='timedelta64[D]').astype('timedelta64[ns]'),
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
            close_dates = np.arange(
                time, 
                time + np.timedelta64(min_days_between_events+1, 'D'), 
                dtype='datetime64[D]')[1:]
            days = days[np.isin(days, close_dates, invert=True)]
    return days

def store_sharc_composite_ds(
    start_year, end_year, base_path, exp_name, 
    min_precip, ar_day, ar_exp=None, precip_var='pr', winter=False):
    data = xr.concat(
        [du.load_daily_sharc_data(base_path, exp_name, year) 
         for year in range(start_year, end_year+1)],
        dim='time')
    data['pr'] = data.lprec + data.fprec
    if ar_day & (ar_exp is not None):
        ar_data = xr.open_mfdataset(
            f'/archive/Marc.Prange/clearwater_data/{ar_exp}/'
            f'{ar_exp}_clearwater_*.nc').sel(
                time=data.time - np.timedelta64(12, 'h'))
        data = data.assign(
            {
                'ar_shape': (('time'), ar_data.ar_shape.values),
                'time': ar_data.time,
                })
    precip_days = du.get_strong_precip_days(
        data, min_precip, ar_day, precip_var, winter) 
    comp_data = create_temporal_composite_ds(data, precip_days)
    dir = f'{base_path}/{exp_name}/precip_composite/'
    Path(dir).mkdir(parents=True, exist_ok=True)
    if ar_day:
        ar_str = 'AR_days'
    else:
        ar_str = 'all_days'
    comp_data.to_netcdf(
        dir+
        f'{exp_name}_{ar_str}_min_{precip_var}_{min_precip}_'
        f'{start_year}-{end_year}_temporal_composite.nc')
    
def load_sharc_composite_ds(
    start_year, end_year, base_path, exp_name, 
    min_precip, ar_str, precip_var='pr'):
    return xr.open_dataset(
        f'{base_path}/{exp_name}/precip_composite/'
        f'{exp_name}_{ar_str}_min_{precip_var}_{min_precip}_'
        f'{start_year}-{end_year}_temporal_composite.nc')  
    
def store_na_composite_mean_ds(
    data, exp_name, out_base_path, min_precip, precip_var, 
    ar_day, winter, queue=None, lon_min=None, lon_max=None):
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
            f'{exp_name}_na_{ar_str}_min_{precip_var}_{min_precip}_'
            f'1990-2020_temporal_composite_mean.nc',)
            # encoding={'mrro_nfpr_frac_bins': {'dtype': 'str'}})
            
def store_loc_composite_ds(
    start_year, end_year, loc_name, 
    base_path, exp_name, min_precip, ar_day):
    comp_data = create_loc_composite_ds(
        start_year, end_year, loc_name, 
        base_path, exp_name, min_precip, ar_day)
    if ar_day:
        ar_str = 'AR_days'
    else:
        ar_str = 'all_days'
    dir = f'{base_path}/{loc_name}_data/{exp_name}/precip_composite/'
    Path(dir).mkdir(parents=True, exist_ok=True)
    comp_data.to_netcdf(
        dir+f'{exp_name}_{loc_name}_{ar_str}_min_precip_'
        f'{min_precip}_{start_year}-{end_year}_temporal_composite.nc')

def get_strong_precip_days(
    data, min_precip=30, ar_day=True, precip_var='pr', winter=False):
    filter = (data[f'{precip_var}'] > min_precip/86400)
    if ar_day:
        filter &= (data.ar_shape == 1)
    if winter:
        print("Filtering for winter events...", flush=True)
        data = data.isel(
            time=np.isin(data['time.month'], [11, 12, 1, 2, 3]))
    strong_precip_days = data.time.where(filter).dropna('time')
    return strong_precip_days

def create_loc_composite_ds(
    start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_day):
    data = du.load_loc_data(start_year, end_year, exp_name, loc_name, base_path)
    precip_days = get_strong_precip_days(data, min_precip, ar_day)
    comp_data = create_temporal_composite_ds(data, precip_days)
    return comp_data

def load_loc_composite_ds(
    start_year, end_year, loc_name, base_path, exp_name, min_precip, ar_str):
    return xr.open_dataset(
        f'{base_path}/{loc_name}_data/{exp_name}/precip_composite/'
        f'{exp_name}_{loc_name}_{ar_str}_min_precip_'
        f'{min_precip}_{start_year}-{end_year}_temporal_composite.nc'
    )

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

def temporal_composite_ds_mean_for_precip(
    data_loc, min_precip=20, precip_var='pr', ar_day=False, winter=False):
    precip_days = get_strong_precip_days(
        data_loc, min_precip, ar_day, precip_var=precip_var, winter=winter)
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
            f'{exp_name}_na_{ar_str}_min_{precip_var}_{min_precip}_'
            f'1990-2020_temporal_composite_mean_{winter_str}_lon_'
            f'{np.round(lon_min, 2)}-{np.round(lon_max, 2)}.nc',)