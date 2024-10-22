import xarray as xr
from glob import glob
import argparse
# from dask.distributed import Client


def imerg_hd5_to_netcdf(
        year,
        month,
        base_path='/uda/GPM_IMERG', 
        out_path='/archive/Marc.Prange/IMERG/netcdf_raw', 
        ):
    # client = Client(n_workers=20, threads_per_worker=2, memory_limit='10GB')
    files = glob(
        f'{base_path}/{year}/3B-HHR.MS.MRG.3IMERG.{year}{month:02}*-S*-E*.*.V07B.HDF5')
    if files == []:    
        print(f"No IMERG HDF files found for {year}/{month:02}.", flush=True)
        return
    print(f"Opening IMERG HDF data for {year}/{month:02}.", flush=True)
    data = xr.open_mfdataset(
        files, group='Grid', combine="nested", 
        concat_dim="time", parallel=True, engine='h5netcdf')
    print(f"Storing IMERG data for {year}/{month:02} as NetCDF.", flush=True)
    data.to_netcdf(
        f'{out_path}/IMERG_raw_{year}_{month:02}.V07B.nc'
    )

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2000)
    parser.add_argument("--month", type=int, default=1)
    args = parser.parse_args()
    imerg_hd5_to_netcdf(args.year, args.month)

if __name__ == '__main__':
    _main()
