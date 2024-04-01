import xarray as xr

data = xr.open_dataset(
    '/archive/Ming.Zhao/awg/2022.03/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
    'gfdl.ncrc4-intel-prod-openmp/pp/atmos_cmip/ts/daily/1yr/atmos_cmip.20160101-20161231.prw.nc')
print(data)