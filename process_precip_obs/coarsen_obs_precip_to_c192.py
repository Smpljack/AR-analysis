import xarray as xr
import numpy as np
from scipy.interpolate import griddata

import data_util as du


def coarsen_precipitation(obs_file, c192_file, output_file):
    # Load the observational precipitation dataset
    obs_ds = xr.open_dataset(obs_file, engine='h5netcdf')
    
    # Load the c192 model grid
    c192_ds = xr.open_dataset(c192_file)
    
    # Extract precipitation data and coordinates
    precip = obs_ds['precipitation']
    precip = du.lon_180_to_360(precip)
    lat = obs_ds['lat']
    lon = obs_ds['lon']
    
    # Extract c192 grid coordinates
    c192_lat = c192_ds['lat']
    c192_lon = c192_ds['lon']
    
    # Calculate the resolution difference
    res_factor = int((c192_lat[1] - c192_lat[0]) / (lat[1] - lat[0]))
    
    # Perform coarse graining using rolling mean
    coarse_precip = precip.rolling(lat=res_factor, lon=res_factor, center=True).mean()
     
    # Interpolate to c192 grid
    precip_interp = griddata(
        (coarse_precip['lon'].values.flatten(), coarse_precip['lat'].values.flatten()),
        coarse_precip.values.flatten(),
        (c192_lon.values[None, :], c192_lat.values[:, None]),
        method='linear'
    )
    
    # Create a new dataset with coarse-grained precipitation
    coarse_ds = xr.Dataset(
        {
            'precipitation': (['lat', 'lon'], precip_interp),
        },
        coords={
            'lat': c192_lat,
            'lon': c192_lon,
        }
    )
    
    # Add metadata
    coarse_ds['precipitation'].attrs = precip.attrs
    coarse_ds.attrs['description'] = f'Coarse-grained precipitation data from {obs_file} to c192 grid resolution'
    
    # Save the coarse-grained dataset
    coarse_ds.to_netcdf(output_file)
    print(f"Coarse-grained precipitation saved to {output_file}")

# Example usage
obs_file = '/archive/Marc.Prange/IMERG/netcdf_raw/IMERG_raw_2021_10.V07B.nc'
c192_file = '/archive/Ming.Zhao/awg/2022.03/c192_obs/atmos_data/daily_mswep/mswep.19810101-19811231.precipitation.nc'
output_file = '/archive/Marc.Prange/imerg/netcdf_raw/coarse_grained_precipitation.nc'

coarsen_precipitation(obs_file, c192_file, output_file)

