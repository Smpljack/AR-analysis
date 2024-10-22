import xarray as xr
import numpy as np
from pathlib import Path
import argparse
from glob import glob
import pandas as pd
from global_land_mask import globe
import regionmask


import comp_event_util as ceu


exp_basepath_map = {
        'c192L33_am4p0_amip_HIRESMIP_HX': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_HX_p2K': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K': '/archive/Ming.Zhao/awg/2023.04/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min': '/archive/Ming.Zhao/awg/2023.04/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K': '/archive/Ming.Zhao/awg/2023.04/',
    }


def lon_360_to_180(ds):
    """
    Converts longitudes from 0-360 to -180 to 180.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray with converted longitudes.
    """
    ds['lon'] = xr.where(ds.lon > 180, ds.lon - 360, ds.lon)
    ds = ds.sortby('lon')
    return ds

def lon_180_to_360(ds):
    """
    Converts longitudes from -180-180 to 0-360.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray with converted longitudes.
    """
    ds['lon'] = xr.where(ds.lon < 0, ds.lon + 360, ds.lon)
    ds = ds.sortby('lon')
    ds['lon'] = ds['lon'].assign_attrs(units="degrees_east", axis="X", long_name='longitude', bounds='lon_bnds')
    return ds

def sel_na_pacific(ds):
    """
    Selects data within the North Pacific region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray containing data within the North Pacific region.
    """
    return ds.sel({'lat': slice(10, 60), 'lon': slice(-170, -80)})

def sel_na_westcoast(ds):
    """
    Selects data within the North American West Coast region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray containing data within the North American West Coast region.
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
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray containing data within the California region.
    """
    return ds.sel({'lat': slice(31, 44), 'lon': slice(-130, -110)})

def sel_na(ds):
    """
    Selects data within the broader North America region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray containing data within the broader North America region.
    """
    return ds.sel({'lat': slice(20, 70), 'lon': slice(-140, -60)})

def sel_conus_land(ds):
    """
    Selects data within the broader North America region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with latitude and longitude coordinates.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray containing data within the broader North America region.
    """
    ds = ds.sel({'lat': slice(25, 50), 'lon': slice(-125, -60)})
    # lat, lon = np.meshgrid(ds.lat, ds.lon)
    # land_mask = xr.DataArray(
    #             globe.is_land(lat, lon).T,
    #             coords={'lat': ds.lat, 'lon': ds.lon}
    #         )
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    states_mask = states.mask(ds.lon, ds.lat, wrap_lon=False)
    return ds.where(~np.isnan(states_mask))

def sel_region(ds, region):
    """
    Selects data for a specified region.

    Parameters:
    - ds (xarray.Dataset or xarray.DataArray):
      Input dataset with latitude and longitude coordinates.
    - region (str):
      Name of the region to select. Valid options are 'california', 'na', 'conus_land', and 'global'.

    Returns:
    - xarray.Dataset or xarray.DataArray:
      Dataset or DataArray containing data for the specified region.
    """
    if region == 'california':
        return sel_california(ds)
    elif region == 'na':
        return sel_na(ds)
    elif region == 'conus_land':
        return sel_conus_land(ds)
    elif region == 'global':
        return ds  # Return the original dataset for global selection
    else:
        raise ValueError(f"Invalid region: {region}. Valid options are 'california', 'na', 'conus_land', and 'global'.")


def preprocess_ar_ds(ds):
    return ds.drop_vars(['islnd', 'iscst', 'axis', 'lfloc'])
  
def load_ar_data(base_path, exp_name, year, daily_data=True):
    """
    Loads Atmospheric River (AR) data.

    Parameters:
    - base_path (str): Base path for the data files.
    - exp_name (str): Experiment name.
    - year (int): Year of the data.

    Returns:
    - xarray.Dataset: AR data for the specified year.
    """
    ar_data = xr.open_mfdataset(
            base_path+f'{exp_name}/AR_climlmt/{exp_name}_AR_{year}.nc', 
            parallel=False, concat_dim="time", combine="nested",
            data_vars='minimal', coords='minimal', compat='override',
            preprocess=preprocess_ar_ds).load()
    return ar_data

def load_daily_ar_shape(exp_name, year):
    """
    Loads daily AR shape data.

    Parameters:
    - base_path (str): Base path for the data files.
    - exp_name (str): Experiment name.
    - year (int): Year of the data.

    Returns:
    - xarray.Dataset: Daily AR shape data for the specified year.
    """
    ar_shape = xr.open_mfdataset(
        f'/archive/Marc.Prange/ar_shape/{exp_name}/{exp_name}_daily_AR_shape_{year}.nc', 
        concat_dim='time', combine='nested')
    return ar_shape

def load_mswep_pr_data(base_path, exp_name, year):
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
        base_path+f'{exp_name}/atmos_data/daily_mswep/'
        f'mswep.{year}0101-{year}1231.precipitation.nc')
    pr_data['precipitation'] = pr_data.precipitation / 86400
    return pr_data

def load_gpm_pr_data(year):
    pr_data = xr.open_dataset(
        f'/archive/Marc.Prange/GPM/daily/GPM_{year}.c192_daily.nc') 
    pr_data = pr_data.rename({'precip': 'precipitation'})
    pr_data['precipitation'] = pr_data.precipitation / 3600
    return pr_data

def load_stageiv_pr_data(year):
    pr_data = xr.open_mfdataset(
        f'/archive/Marc.Prange/StageIV/daily/StageIV_*_{year}.c192_daily.nc')
    pr_data = pr_data.rename({'p01m': 'precipitation'})
    pr_data['precipitation'] = pr_data.precipitation / 3600
    return pr_data

def load_imerg_pr_data(year):
    pr_data = xr.open_mfdataset(
        f'/archive/Marc.Prange/IMERG/daily/IMERG_*_{year}.c192_daily.nc',
        combine='nested', concat_dim='time')
    pr_data['precipitation'] = pr_data.precipitation / 3600
    return pr_data

def load_model_data(
    base_path, year, variables,
    exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
    ar_analysis=True,
    min_ar_pr_threshold=1/86400,
    low_pr_value=np.nan,
    min_pr_var='precip',
    ar_masked_vars=['precip'],
    gfdl_processor='gfdl.ncrc4-intel-prod-openmp',
    lon_180=True):
    """
    Loads model data.

    Parameters:
    - base_path (str):
      Base path for the data files.
    - year (int):
      Year of the data.
    - variables (list):
      List of variables to load.
    - exp_name (str):
      Experiment name. Default is 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020'.
    - ar_analysis (bool):
      Whether to perform Atmospheric River (AR) analysis. Default is True.
    - gfdl_processor (str):
      Processor name. Default is 'gfdl.ncrc4-intel-prod-openmp'.

    Returns:
    - xarray.Dataset: Model data for the specified year and variables.
    """
    print(f'Loading model data for year {year}\nexp_name: {exp_name}', flush=True)
    cmip_atmos_vars = ['pr', 'prsn', 'ts', 't_ref', 'prw']
    cmip_land_vars = ['snw', 'snc', 'mrro', 'mrso', 'mrsos', 'snm', 'tran']
    land_vars = ['evap_land', 'npp', 'precip', 'runf_soil', 'runf_lake',
                 'soil_ice', 'soil_liq', 't_ref_max', 't_ref_min',
                 'vegn_temp_max', 'vegn_temp_min']
    river_variables = ['rv_d_h2o', 'rv_o_h2o']
    if ar_analysis and min_pr_var not in variables:
        variables.append(min_pr_var)
    ds_list = []
    if np.isin(variables, cmip_atmos_vars).any():
        print('Loading CMIP atmos data...', flush=True)
        cmip_atmos_data = xr.merge(
            [xr.open_mfdataset(
                base_path+f'{exp_name}/{gfdl_processor}/pp/atmos_cmip/ts/daily/1yr/'
                f'atmos_cmip.{year}0101-{year}1231.{var}.nc',
                parallel=False, concat_dim="time", combine="nested",
                data_vars='minimal', coords='minimal', compat='override')
                for var in variables
                if var not in ['ivtx', 'ivty']
                and var not in land_vars
                and var not in cmip_land_vars
                and var not in river_variables])
        ds_list.append(cmip_atmos_data)
    if np.isin(variables, cmip_land_vars).any():
        print('Loading CMIP land data...', flush=True)  
        cmip_land_data = xr.merge(
                [xr.open_mfdataset(
                    base_path+f'{exp_name}/{gfdl_processor}/pp/land_cmip/ts/daily/1yr/'
                    f'land_cmip.{year}0101-{year}1231.{var}.nc',
                    parallel=False, concat_dim="time", combine="nested",
                    data_vars='minimal', coords='minimal', compat='override')
                 for var in variables if var in cmip_land_vars])
        ds_list.append(cmip_land_data)
    if np.isin(variables, land_vars).any():
        print('Loading land data...', flush=True)
        land_data = xr.merge(
                [xr.open_mfdataset(
                    base_path+f'{exp_name}/{gfdl_processor}/pp/land/ts/daily/1yr/'
                    f'land.{year}0101-{year}1231.{var}.nc',
                    parallel=True, concat_dim="time", combine="nested",
                    data_vars='minimal', coords='minimal', compat='override')
                 for var in variables if var in land_vars])
        ds_list.append(land_data)
    if np.isin(variables, river_variables).any():
        print('Loading river data...', flush=True)
        river_data = xr.merge(
            [xr.open_mfdataset(base_path+f'{exp_name}/{gfdl_processor}/pp/river/ts/daily/1yr/'
            f'river.{year}0101-{year}1231.{var}.nc',
            parallel=True, concat_dim="time", combine="nested",
            data_vars='minimal', coords='minimal', compat='override'    )
             for var in variables if var in river_variables])
        ds_list.append(river_data)
    data = xr.merge(ds_list)
    if 'prli' in variables:
        data['prli'] = data.pr - data.prsn
    data = lon_360_to_180(data) if lon_180 else data
    data['time'] = np.array(data.time.values, dtype='datetime64[1D]')
    if 'ivtx' in variables:
        ivt_data = xr.merge(
            [xr.open_dataset(base_path+f'{exp_name}/{gfdl_processor}/pp/atmos_cmip/ts/6hr/1yr/'
            f'atmos_cmip.{year}010100-{year}123123.{var}.nc')
             for var in ['ivtx', 'ivty']]).resample({'time': '1D'}).mean()
        ivt_data = lon_360_to_180(ivt_data) if lon_180 else ivt_data
        ivt_data = ivt_data.assign_coords({'lat': data.lat, 'lon': data.lon})
        ivt_data['time'] = np.array(ivt_data.time.values, dtype='datetime64[1D]')
        ivt_data = ivt_data.isel(time=np.isin(ivt_data.time, data.time))
    if ar_analysis:
        print('Loading AR data...', flush=True)
        ar_shape = load_daily_ar_shape(exp_name, year)
        ar_shape = lon_360_to_180(ar_shape) if lon_180 else ar_shape
        ar_shape = ar_shape.assign_coords({'lat': data.lat, 'lon': data.lon})
        data['ar_shape'] = ar_shape.shape.load()
        print('Masking AR data...', flush=True)
        data = ar_masked(data.load(), ar_masked_vars, min_ar_pr_threshold, min_pr_var, low_pr_value)
    return data


def load_obs_data(
        base_path, year, era5_variables=None, exp_name='c192_obs', ar_analysis=True, obs_pr_dataset=None,
        min_ar_pr_threshold=1/86400,
        low_pr_value=np.nan,
        min_pr_var='precip',
        ar_masked_vars=['precip'], 
        lon_180=True
        ):
    """
    Loads ERA5 data.

    Parameters:
    - base_path (str):
      Base path for the data files.
    - year (int):
      Year of the data.
    - variables (list):
      List of variables to load.
    - ar_analysis (bool):
      Whether to perform Atmospheric River (AR) analysis. Default is True.
    - mswep_precip (bool):
      Whether to load MSWEP precipitation data. Default is True.
    - exp_name (str):
      Experiment name. Default is 'c192_obs'.

    Returns:
    - xarray.Dataset: ERA5 data for the specified year and variables.
    """
    if era5_variables is not None:
        era5_data = xr.merge(
            [xr.open_dataset(
                base_path+
                f'{exp_name}/atmos_data/daily_era5/ERA5.{year}0101-{year}1231.{var}.nc') for var in era5_variables])
        era5_data = lon_360_to_180(era5_data) if lon_180 else era5_data
        era5_data['time'] = np.array(era5_data.time.values, dtype='datetime64[1D]')
    else:
        era5_data = None
    if obs_pr_dataset is not None:
        if obs_pr_dataset == 'mswep':
            obs_pr_data = load_mswep_pr_data(base_path, exp_name, year)
        elif obs_pr_dataset == 'gpm':
            obs_pr_data = load_gpm_pr_data(year)
        elif obs_pr_dataset == 'stageiv':
            obs_pr_data = load_stageiv_pr_data(year)
        elif obs_pr_dataset == 'imerg':
            obs_pr_data = load_imerg_pr_data(year)
        obs_pr_data = lon_360_to_180(obs_pr_data) if lon_180 else obs_pr_data
        obs_pr_data['time'] = np.array(obs_pr_data.time.values, dtype='datetime64[1D]')
    else: 
        obs_pr_data = None
    data = xr.merge(
        [data for data in [era5_data, obs_pr_data] if data is not None], 
        compat='override').rename({'precipitation': 'precip'})
    if ar_analysis:
        print('Loading AR data...', flush=True)
        ar_shape = load_daily_ar_shape(exp_name, year)
        ar_shape = lon_360_to_180(ar_shape) if lon_180 else ar_shape
        ar_shape = ar_shape.assign_coords({'lat': data.lat, 'lon': data.lon})
        data['ar_shape'] = ar_shape.shape.load()
        print('Masking AR data...', flush=True)
        data = ar_masked(data.load(), ar_masked_vars, min_ar_pr_threshold, min_pr_var, low_pr_value)
    return data

def load_daily_sharc_data(
    base_path='/archive/Marc.Prange/LM4p2_SHARC',
    exp_name='providence_lm4sharc_ksat0016_angle087rad_ep26_114y',
    year='all_years'):
    """
    Loads daily SHARC data.

    Parameters:
    - base_path (str):
      Base path for the data files. Default is '/archive/Marc.Prange/LM4p2_SHARC'.
    - exp_name (str):
      Experiment name. Default is 'providence_lm4sharc_ksat0016_angle087rad_ep26_114y'.
    - year (int or str):
      Year of the data. Default is 'all_years'.

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

def daily_ar_shape(ar_shape, days):
    """
    Maps 6hourly AR shape data to daily AR data.
    If any point in 6hourly is AR on given day, point is flagged as AR day.
    Parameters:
    - ar_shape (xarray.Dataset): AR shape data.
    - days (array-like): Array-like object containing days.

    Returns:
    - xarray.DataArray: Daily AR shape data.
    """

    # Create a DataArray with the same dimensions as the input, but for daily data
    daily_ar_shape = xr.DataArray(
        coords={
            'time': days,
            'lat': ar_shape.lat,
            'lon': ar_shape.lon,
        },
        dims=['time', 'lat', 'lon']
    )

    # Group the 6-hourly data by day and sum over each day
    daily_sum = ar_shape.resample(time="1D").sum()

    # Assign True where sum > 0, and NaN otherwise
    daily_ar_shape.values = xr.where(daily_sum > 0, True, np.nan).values

    return daily_ar_shape

def ar_masked(data, masked_vars='pr', min_pr_threshold=1, min_pr_var='pr', low_pr_value=np.nan):
    """
    Mask precipitation field by detected ARs.

    Args:
        data (xr.Dataset): Dataset containing precipitation variable
        pr_var (str, optional): Precipitation variable to mask. Defaults to 'pr'.

    Returns:
        xr.DataArray: Returns DataArray where non-AR pixels are NaN.
    """
    for var in masked_vars:
        data[f'ar_{var}'] = data[var].where(
            (data.ar_shape == 1) & 
            (data[f'{min_pr_var}'] > min_pr_threshold),
            other=low_pr_value)
    return data


def store_yearly_NA_model_data(
        year,
        base_path='/archive/Ming.Zhao/awg/2022.03/',
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
        out_base_path='/archive/Marc.Prange/',
        gfdl_processor='gfdl.ncrc4-intel-prod-openmp',
        variables=['ts', 'prw', 'pr', 'prsn', 'ivtx', 'ivty', 'wap500', 'mrro', 'mrsos',
                   'mrso', 'snw', 'evap_land', 'precip', 'rv_d_h2o', 'rv_o_h2o'],
        ar_analysis=True, min_ar_pr_threshold=1/86400, low_pr_value=np.nan, 
        min_pr_var='precip', ar_masked_vars=['precip']):
    """
    Store North American model output data for one year.
    Args:
        year (int): Year to store.
        base_path (str, optional): Path leading to directory containing location name.
                                   Defaults to '/archive/Ming.Zhao/awg/2022.03/'.
        exp_name (str, optional): Experiment name.
                                  Defaults to 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020'.
        out_base_path (str, optional): Path to directory for storing the data.
                                       Defaults to '/archive/Marc.Prange/'.
        gfdl_processor (str, optional): Name of GFDL post-processor used.
                                        Defaults to 'gfdl.ncrc4-intel-prod-openmp'.
        variables (list, optional): List of variables to store.
                                    Defaults to ['ts', 'prw', 'pr', 'prsn', 'ivtx', 'ivty',
                                                 'mrro', 'mrsos', 'mrso', 'snw', 'evap_land',
                                                 'precip'].
        ar_analysis (bool, optional): Whether to store AR-shape variable from
                                      Guan and Waliser (2015) AR detection. Defaults to True.
    """
    print(f'Loading model data for {year}...', flush=True)
    model_data = load_model_data(
        base_path, year,
        exp_name=exp_name,
        gfdl_processor=gfdl_processor,
        variables=variables, ar_analysis=ar_analysis, 
        min_ar_pr_threshold=min_ar_pr_threshold, low_pr_value=low_pr_value, 
        min_pr_var=min_pr_var, ar_masked_vars=ar_masked_vars)
    model_data_na = sel_na(model_data)
    print(f'Storing NA data for {year}...', flush=True)
    dir = f'{out_base_path}/na_data/{exp_name}/'
    Path(dir).mkdir(parents=True, exist_ok=True)
    model_data_na.to_netcdf(
        f'{dir}{exp_name}_na_{year}.nc')

def store_yearly_NA_obs_data(
        year,
        base_path='/archive/Ming.Zhao/awg/2022.03/',
        exp_name='c192_obs',
        out_base_path='/archive/Marc.Prange/',
        era5_variables=None,
        ar_analysis=True, obs_pr_dataset='mswep'):
    """
    Store North American observational data for one year.

    Args:
        year (int): Year to store.
        base_path (str, optional): Path leading to directory containing location name.
                                   Defaults to '/archive/Ming.Zhao/awg/2022.03/'.
        exp_name (str, optional): Experiment name. Defaults to 'c192_obs'.
        out_base_path (str, optional): Path to directory for storing the data.
                                       Defaults to '/archive/Marc.Prange/'.
        variables (list, optional): List of variables to store.
                                    Defaults to ['prw', 'ivtx', 'ivty'].
        ar_analysis (bool, optional): Whether to store AR-shape variable from
                                      Guan and Waliser (2015) AR detection.
                                      Defaults to True.
        mswep_precip (bool, optional): Whether to store precipitation from
                                       MSWEP dataset. Defaults to True.
    """
    print(f'Loading obs data for {year}...', flush=True)
    obs_data = load_obs_data(
        base_path, year,
        era5_variables=era5_variables, obs_pr_dataset=obs_pr_dataset, 
        exp_name=exp_name, ar_analysis=ar_analysis)
    obs_data_na = sel_na(obs_data)
    print(f'Storing NA data for {year}...', flush=True)
    dir = f'{out_base_path}/na_data/{exp_name}/'
    if obs_pr_dataset is None:
        file = f'{exp_name}_na_{year}.nc'
    else: 
        file = f'{obs_pr_dataset}.{exp_name}_na_{year}.nc'
    Path(dir).mkdir(parents=True, exist_ok=True)
    obs_data_na.to_netcdf(dir+file)



def load_loc_data(start_year, end_year, exp_name, loc_name, base_path='/archive/Marc.Prange/'):
    """
    Load dataset from a specific location.

    Args:
        start_year (int): First year of data to be loaded.
        end_year (int): Last year of data to be loaded.
        exp_name (str): Experiment name.
        loc_name (str): Location name.
        base_path (str, optional): Path leading to directory containing location name.
                                   Defaults to '/archive/Marc.Prange/'.

    Returns:
        xr.Dataset: Local dataset
    """
    print(f'Loading loc_data for {loc_name}', flush=True)
    return xr.concat([xr.open_dataset(
        f'{base_path}{loc_name}_data/'
        f'{exp_name}/{exp_name}_{loc_name}_{year}.nc')
        for year in range(start_year, end_year+1)], dim='time')

def interp_data(data_array, delta_xy_deg=0.25):
    """
    Interpolate data_array (model/obs) to new resolution.

    Args:
        data_array (xr.DataArray): DataArray containing data on a lat/lon grid.
        delta_xy_deg (float, optional): New resolution in degree. Defaults to 0.25.
    """
    lat_fine = np.arange(data_array.lat.min(), data_array.lat.max()+delta_xy_deg, delta_xy_deg)
    lon_fine = np.arange(data_array.lon.min(), data_array.lon.max()+delta_xy_deg, delta_xy_deg)
    return data_array.interp({'lon': lon_fine, 'lat': lat_fine})

def load_monthly_ts_all(exp_name, base_path, var_type='2d', ):
    vars_3d = ['ucomp', 'vcomp', 'sphum', 'cld_amt', 'liq_wat', 
               'ice_wat', 'liq_drp', 'aliq', 'omega', 'hght', 'temp']
    if var_type == '2d':
        paths = [
            path for path in glob(f'{base_path}{exp_name}/ts_all/atmos.*.nc')
            if ~np.any([var in path for var in vars_3d])]
    elif var_type == '3d':
        paths = glob(f'{base_path}{exp_name}/ts_all/atmos.*.nc')
    data = xr.open_mfdataset(paths)
    return data

def store_monthly_mean_ar_masked_data(
            exp_name, base_path, variables, start_year, end_year, 
            gfdl_processor='gfdl.ncrc4-intel-prod-openmp',
            outpath='/archive/Marc.Prange/ar_masked_monthly_data/',
            min_precip=None):
    print("Storing monthly mean AR masked data for:", flush=True)
    for year in range(start_year, end_year+1):
        print(f"\n{year}\t{exp_name}", flush=True)
        ar_data = load_ar_data(base_path, exp_name, year).load()
        ar_data = lon_360_to_180(ar_data)
        model_data = load_model_data(
            base_path, year, variables, exp_name, 
            ar_analysis=False, gfdl_processor=gfdl_processor).load()
        ar_data['lon'] = model_data.lon
        ar_data['lat'] = model_data.lat
        ar_shape = daily_ar_shape(ar_data, model_data.time)
        ar_shape = ~np.isnan(ar_shape)
        if min_precip is not None:
            ar_shape &= (model_data.pr*86400 > min_precip)
        model_data = model_data.where(ar_shape)
        model_data_monthly_mean = model_data.groupby('time.month').mean()
        model_data_monthly_sum = model_data.groupby('time.month').sum()
        model_data_monthly_std = model_data.drop(
            'average_DT', errors='ignore').groupby('time.month').std()
        model_data_monthly_max = model_data.groupby('time.month').max()
        ar_count_monthly = ar_shape.groupby('time.month').sum()
        model_data_monthly_mean = model_data_monthly_mean.assign_coords(
            time=(('month'), np.arange(f'{year}-01', f'{year+1}-01', dtype='datetime64[M]'))
            ).swap_dims({'month': 'time'})
        model_data_monthly_sum = model_data_monthly_sum.assign_coords(
        time=(('month'), np.arange(f'{year}-01', f'{year+1}-01', dtype='datetime64[M]'))
            ).swap_dims({'month': 'time'})
        model_data_monthly_std = model_data_monthly_std.assign_coords(
            time=(('month'), np.arange(f'{year}-01', f'{year+1}-01', dtype='datetime64[M]'))
            ).swap_dims({'month': 'time'})
        model_data_monthly_max = model_data_monthly_max.assign_coords(
            time=(('month'), np.arange(f'{year}-01', f'{year+1}-01', dtype='datetime64[M]'))
            ).swap_dims({'month': 'time'})
        ar_count_monthly = ar_count_monthly.assign_coords(
            time=(('month'), np.arange(f'{year}-01', f'{year+1}-01', dtype='datetime64[M]'))
            ).swap_dims({'month': 'time'}).rename('ar_count')
        ar_count_monthly.to_netcdf(
                f'{outpath}{exp_name}/{exp_name}_AR_count_monthly_min_precip_{str(min_precip)}.{year}.nc'
            )
        for var in variables + ['prli']:
            model_data_monthly_mean[var].to_netcdf(
                f'{outpath}{exp_name}/{exp_name}_AR_masked_min_precip_{str(min_precip)}_monthly_mean.{year}.{var}.nc'
            )
            model_data_monthly_sum[var].to_netcdf(
                f'{outpath}{exp_name}/{exp_name}_AR_masked_min_precip_{str(min_precip)}_monthly_sum.{year}.{var}.nc'
            )
            model_data_monthly_std[var].to_netcdf(
                f'{outpath}{exp_name}/{exp_name}_AR_masked_min_precip_{str(min_precip)}_monthly_std.{year}.{var}.nc'
            )
            model_data_monthly_max[var].to_netcdf(
                f'{outpath}{exp_name}/{exp_name}_AR_masked_min_precip_{str(min_precip)}_monthly_max.{year}.{var}.nc'
            )
            
def load_ar_day_avg_stat(exp_name, base_path, start_year, end_year, stat='mean', min_precip=1, var='*'):
    if stat == 'count':
        ar_count_paths = np.sort(
            glob(f'{base_path}{exp_name}/{exp_name}_AR_count_monthly_min_precip_{str(min_precip)}.*.nc'))
        ar_count_monthly = xr.open_mfdataset(
            ar_count_paths, coords='minimal').sel(time=slice(f'{start_year}', f'{end_year}')).ar_count.load()
        return ar_count_monthly
    ar_masked_paths = np.sort(
        glob(f'{base_path}{exp_name}/{exp_name}_AR_masked_min_precip_{str(min_precip)}_monthly_{stat}.*.{var}.nc'))
    ar_masked_monthly = xr.open_mfdataset(
        ar_masked_paths, coords='minimal').sel(time=slice(f'{start_year}', f'{end_year}')).load()
    return ar_masked_monthly

def std_long_from_std_monthly(std_monthly, N_monthly, mean_monthly):
    """Calculate longterm 'mean' standard deviation from monthly stds.

    Args:
        std_monthly (xr.DataArray): Monthly std
        N_monthly (xr.DataArray): Monthly sample size
        mean_monthly (xr.DataArray): Monthly mean

    Returns:
        xr.DataArray: Long-term standard deviation
    """
    mean_long = (mean_monthly * N_monthly).sum('time') / N_monthly.sum('time')
    std_long = np.sqrt(
        ((std_monthly**2*(N_monthly-1)) + N_monthly*(mean_long - mean_monthly)**2).sum('time') / 
                (N_monthly.sum('time') - 1))
    return std_long

def std_long_from_std_monthly_for_ds(ds):
    """Calculate longterm 'mean' standard deviation from monthly stds.

    Args:
        std_monthly (xr.DataSet): Contains three variables
            1.) std: Monthly standard deviations.
            2.) count: Monthly number of samples.
            3.) mean: Monthly mean.
    Returns:
        xr.DataArray: Long-term standard deviation
    """
    mean_long = (ds['mean'] * ds['count']).sum('time') / ds['count'].sum('time')
    std_long = np.sqrt(
        ((ds['std']**2*(ds['count']-1)) + ds['count']*(mean_long - ds['mean'])**2).sum('time') / 
                (ds['count'].sum('time') - 1))
    return std_long

def store_ar_day_means(
    exp_name, start_year, end_year, 
    variables=['ts', 'prw', 'pr', 'prsn', 'prli','ivtx', 'ivty', 'wap500', 'mrro', 'mrsos',
               'mrso', 'snw', 'evap_land', 'precip', 'rv_d_h2o', 'rv_o_h2o'],
    base_path='/archive/Marc.Prange/ar_masked_monthly_data/', 
    min_precip=1, monthly_means=True):
    print(f"Storing AR count for {start_year}-{end_year}", flush=True)
    data_ar_count = load_ar_day_avg_stat(
        exp_name, base_path, start_year, end_year, stat='count', min_precip=min_precip)
    data_ar_count.to_netcdf(
        f'{base_path}{exp_name}/ar_day_mean/'
        f'{exp_name}_ar_count_min_precip_{min_precip}'
        f'.{start_year}-{end_year}.nc'
    )
    for var in variables:
        print(f"Storing AR day mean for: {var}\t{start_year}-{end_year}", flush=True)
        data_ar_masked_sum = load_ar_day_avg_stat(
            exp_name, base_path, start_year, end_year, stat='sum', min_precip=min_precip, var=var)
        if not monthly_means:
            data_ctrl_ar_day_mean = (data_ar_masked_sum / data_ar_count).mean('time')
            data_ctrl_ar_masked_std = load_ar_day_avg_stat(
                exp_name, base_path, start_year, end_year, stat='std', min_precip=min_precip, var=var)
            # Get long-term standard deviation
            std_long = std_long_from_std_monthly(
                data_ctrl_ar_masked_std, data_ar_count, data_ar_masked_sum / data_ar_count)
            days_in_month = xr.DataArray(
                name='days_in_month',
                coords={'time': data_ar_masked_sum.time.values}, 
                data=[pd.Period(str(date)).days_in_month for date in data_ar_masked_sum.time.values])
            data_ctrl_ar_all_day_mean = (data_ar_masked_sum / days_in_month).mean('time')
        else:
            data_ctrl_ar_day_mean = (data_ar_masked_sum / data_ar_count).groupby('time.month').mean()
            data_ctrl_ar_masked_std = load_ar_day_avg_stat(
                exp_name, base_path, start_year, end_year, stat='std', min_precip=min_precip, var=var)
            std_long_input_ds = xr.merge(
                [data_ctrl_ar_masked_std.rename({f'{var}': 'std'}), 
                 data_ar_count.rename('count'), 
                 (data_ar_masked_sum/data_ar_count).rename({f'{var}': 'mean'})])
            std_long = std_long_input_ds.groupby('time.month').map(std_long_from_std_monthly_for_ds)
            days_in_month = xr.DataArray(
                name='days_in_month',
                coords={'time': data_ar_masked_sum.time.values}, 
                data=[pd.Period(str(date)).days_in_month for date in data_ar_masked_sum.time.values])
            data_ctrl_ar_all_day_mean = (data_ar_masked_sum / days_in_month).groupby('time.month').mean()
        # Store data
        if monthly_means:
            mean_type = 'monthly_mean'
            std_type = 'monthly_std'
        else:
            mean_type = 'mean'
            std_type = 'std'
        data_ctrl_ar_day_mean.to_netcdf(
            f'{base_path}{exp_name}/ar_day_mean/'
            f'{exp_name}_ar_day_{mean_type}_min_precip_{min_precip}'
            f'.{start_year}-{end_year}.{var}.nc'
        )
        data_ctrl_ar_all_day_mean.to_netcdf(
            f'{base_path}{exp_name}/ar_all_day_mean/'
            f'{exp_name}_ar_all_day_{mean_type}_min_precip_{min_precip}'
            f'.{start_year}-{end_year}.{var}.nc'
        )
        std_long.to_netcdf(
            f'{base_path}{exp_name}/ar_day_mean/'
            f'{exp_name}_ar_day_{std_type}_min_precip_{min_precip}'
            f'.{start_year}-{end_year}.{var}.nc'
        )

def ar_day_monthly_mean_to_ts_all(
        exp_name, base_path, start_year, end_year, variables, min_precip=1):
    for var in variables:
        monthly_means = xr.open_mfdataset(
                    f'{base_path}{exp_name}/{exp_name}_AR_masked_min_precip_'
                    f'{str(min_precip)}_monthly_mean.*.{var}.nc'
                ).sel(time=slice(start_year, end_year+1))
        monthly_means.to_netcdf(
            '/archive/Marc.Prange/ts_all_missing_vars/'
            f'{exp_name}/'
        )

def get_global_mean_dT(exp_ctrl, exp_dist, start_year, end_year):
    ts_ref = xr.open_mfdataset(
        f'{exp_basepath_map[exp_ctrl]}{exp_ctrl}/ts_all/atmos.*.t_ref.nc').t_ref.sel(
            time=slice(str(start_year), str(end_year))).mean()
    ts_dist = xr.open_mfdataset(
        f'{exp_basepath_map[exp_dist]}{exp_dist}/ts_all/atmos.*.t_ref.nc').t_ref.sel(
            time=slice(str(start_year), str(end_year))).mean()
    return float(ts_dist - ts_ref)

def _main():
    # na_data_obs = xr.open_mfdataset(
    #     '/archive/Marc.Prange/na_data/c192_obs/'
    #     'c192_obs_na_*.nc').load()
    # na_data_model = xr.open_mfdataset(
    #     '/archive/Marc.Prange/na_data/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
    #     'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020_na_*.nc').load()
    # na_data_model_p2K = xr.open_mfdataset(
    #     '/archive/Marc.Prange/na_data/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K/'
    #     'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K_na_*.nc').load()
    # loc_data = na_data_obs.sel({'lon': -123.93, 'lat': 47.72}, method='nearest')
    # precip_days = get_strong_precip_days(loc_data, min_precip=0, ar_day=True, precip_var='pr', winter=False)
    # comp_data = create_temporal_composite_ds(na_data_obs, precip_days, days_back=5, days_ahead=5, min_days_between_events=3)
    # comp_data['loc_lat'] = loc_data.lat
    # comp_data['loc_lon'] = loc_data.lon
    # comp_data.to_netcdf(
    #     '/archive/Marc.Prange/na_data/c192_obs/'
    #     'temporal_composite/clearwater_na_composite_test.nc'
    # )
    # ceu.store_loc_model_data(
    #     start_year=1990, end_year=2020,
    #     loc_lat=47.72, loc_lon=-123.93,
    #     loc_name='clearwater',
    #     model_data=na_data_obs,
    #     base_path='/archive/Marc.Prange/',
    #     exp_name='c192_obs',
    #     out_base_path='/archive/Marc.Prange/',)
    # ceu.store_loc_composite_ds(
    #     start_year=1980, end_year=2014,
    #     loc_name='clearwater',
    #     base_path='/archive/Marc.Prange/',
    #     exp_name='c192_obs',
    #     min_precip=0,
    #     ar_day=True)

    # store_sharc_composite_ds(
    #     start_year=1979, end_year=2014,
    #     base_path='/archive/Marc.Prange/LM4p2_SHARC',
    #     exp_name='clearwater_lm4sharc_ksat002_angle087rad_ep20_114y',
    #     min_precip=0,
    #     ar_day=True,
    #     ar_exp='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
    #     precip_var='pr', winter=False)
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
    # for year in range(1979, 2020):
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_pr_subset", type=str, default='imerg')
    parser.add_argument("--year", type=int, default=2001)

    args = parser.parse_args()
    # store_yearly_NA_obs_data(
    #     args.year,
    #     base_path='/archive/Ming.Zhao/awg/2022.03/',
    #     exp_name='c192_obs',
    #     out_base_path='/archive/Marc.Prange/',
    #     # variables=['prw', 'ivtx', 'ivty'],
    #     ar_analysis=True, obs_pr_dataset=args.obs_pr_subset)
    store_yearly_NA_model_data(
        args.year,
        base_path='/archive/Ming.Zhao/awg/2023.04/',
        exp_name=args.exp_name,
        gfdl_processor='gfdl.ncrc5-intel23-classic-prod-openmp',
        out_base_path='/archive/Marc.Prange/',
        #variables=['prw', 'ivtx', 'ivty'],
        ar_analysis=True)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_name", type=str, default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K')
    # args = parser.parse_args() 
    # store_monthly_mean_ar_masked_data(
    #         exp_name=f'{args.exp_name}', 
    #         base_path='/archive/Ming.Zhao/awg/2023.04/', 
    #         variables=['ts', 'prw', 'pr', 'prsn', 'ivtx', 'ivty', 'wap500',], #'mrro', 'mrsos',
    #                     #'mrso', 'snw', 'evap_land', 'precip', 'rv_d_h2o', 'rv_o_h2o'], 
    #         start_year=1979, end_year=2020, 
    #         gfdl_processor='gfdl.ncrc5-intel23-classic-prod-openmp',
    #         min_precip=1)
    # store_ar_day_means(
    #     exp_name=args.exp_name,
    #     start_year=1980,
    #     end_year=2019,
    #     variables=['pr', 'prsn', 'prli', 'ivtx', 'ivty', 'wap500', 'mrro', 'mrsos',
    #                'mrso', 'snw', 'evap_land', 'precip', 'rv_d_h2o', 'rv_o_h2o'], 
    #     base_path='/archive/Marc.Prange/ar_masked_monthly_data/', 
    #     min_precip=1,
    #     monthly_means=False
    # )
    # base_path = '/archive/Marc.Prange/ar_masked_monthly_data/'
    # exp_name_ctrl = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day'
    # exp_name_p2K = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K'
    # start_year = 1980
    # end_year = 2019
    # data_ctrl_ar_masked = load_ar_day_avg_stat(exp_name_ctrl, base_path, start_year, end_year, stat='mean')
    
if __name__ == '__main__':
    _main()
