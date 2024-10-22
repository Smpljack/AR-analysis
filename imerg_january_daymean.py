# %%
import xarray as xr
import numpy as np
import argparse
# %%
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2001)
args = parser.parse_args()
in_path = f'/archive/Marc.Prange/IMERG/netcdf_raw/IMERG_raw_{args.year}_07.V07B.nc'
out_path = f'/archive/Marc.Prange/IMERG/temp/IMERG_daymean_raw_res_{args.year}_07.V07B.nc'
data = xr.open_dataset(in_path).sortby('time')
# %%
data_daymean = data.resample(time='1D').mean()
data_daymean.to_netcdf(out_path)
