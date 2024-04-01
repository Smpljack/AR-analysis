import os
import numpy as np
import xarray as xr

exp_name = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020'
na_data = xr.open_dataset(
    f'/archive/Marc.Prange/na_data/{exp_name}/{exp_name}_na_2000.nc')
lon_bounds = np.linspace(na_data.lon.min().values, na_data.lon.max().values, 11)
for lon_min, lon_max in zip(lon_bounds[:-1], lon_bounds[1:]):
    os.system("sbatch "
              "/home/Marc.Prange/work/AR-analysis/batch_wrapper.sh "
             f"{lon_min} {lon_max} {exp_name}")
