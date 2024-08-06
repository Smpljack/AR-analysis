import xarray as xr
import numpy as np


def store_monthly_plevel_winds(exp_name, plevels, base_path, out_path):
    u = xr.open_mfdataset(
        f'{base_path}{exp_name}/ts_all/atmos.*.ucomp.nc')
    v = xr.open_mfdataset(
        f'{base_path}{exp_name}/ts_all/atmos.*.vcomp.nc')
    start_year = str(u.time[0].values)[:4]
    end_year = str(u.time[-1].values)[:4]
    for plevel in plevels:
        ulevel = u.sel(level=plevel).rename({'ucomp': f'u_{plevel}'})
        vlevel = v.sel(level=plevel).rename({'vcomp': f'v_{plevel}'})
        ulevel.to_netcdf(
            f'{out_path}{exp_name}/ts_all/atmos.{start_year}01-{end_year}12.u_{plevel}.nc')
        vlevel.to_netcdf(
            f'{out_path}{exp_name}/ts_all/atmos.{start_year}01-{end_year}12.v_{plevel}.nc')



def _main():
    store_monthly_plevel_winds(
        exp_name='c192L33_am4p0_amip_HIRESMIP_HX',
        plevels=[700, 250],
        base_path='/archive/Ming.Zhao/awg/2022.03/',
        out_path='/archive/Marc.Prange/ts_all_upper_winds/'
    )


if __name__ == '__main__':
    _main()