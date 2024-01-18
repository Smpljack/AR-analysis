import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import os
from glob import glob

def lon_360_to_180(ds):
    ds['lon'] = xr.where(ds.lon > 180, ds.lon - 360, ds.lon)
    ds = ds.sortby('lon')
    return ds

def sel_na_westcoast(ds):
     return ds.sel(
                {
                'lat': slice(10, 60),
                'lon': slice(-170, -80)
                }
            )

def load_model_data(base_path, year, variables, ar_analysis=True):
    data = xr.merge(
        [xr.open_dataset(
             base_path+'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
             f'atmos_cmip.{year}0101-{year}1231.{var}.nc') 
             for var in variables if var not in ['ivtx', 'ivty']]).resample(
                  {'time': '1D'}).mean()
    data = lon_360_to_180(data)
    data['time'] = data.indexes['time'].to_datetimeindex()
    if ar_analysis:
        ar_data = xr.open_dataset(
            base_path+'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
            f'AR/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020_AR_{year}.nc').resample(
                {'time': '1D'}
                ).mean()
        ar_data = lon_360_to_180(ar_data)
        ar_data = ar_data.assign_coords({'lat': data.lat, 'lon': data.lon})
        data = xr.merge([data, ar_data])
    if 'ivtx' in variables:
        ivt_data = xr.merge(
            [xr.open_dataset(base_path+f'ERA5/atmos.{year}010100-{year}123123.{var}.nc') 
             for var in ['ivtx', 'ivty']]).resample({'time': '1D'}).mean()
        ivt_data = lon_360_to_180(ivt_data)
        ivt_data = ivt_data.assign_coords({'lat': data.lat, 'lon': data.lon})
        ivt_data['time'] = ivt_data.indexes['time'].to_datetimeindex()
        data = xr.merge([data, ivt_data], compat='override')
    return data


def load_era5_data(base_path, year, variables, ar_analysis=True):
    data = xr.merge(
        [xr.open_dataset(
             base_path+
             f'ERA5/ERA5.{year}010100-{year}123123.{var}.nc') for var in variables]).resample(
    {'time': '1D'}).mean()
    data = lon_360_to_180(data)
    if ar_analysis:
        ar_data = xr.open_dataset(
            base_path+f'ERA5/c192_obs_AR_{year}.nc').resample(
                {'time': '1D'}
                ).mean()
        ar_data = lon_360_to_180(ar_data)
        data = xr.merge([data, ar_data])
    return data

def plot_ivt_bias_rr(model_data, era5_data):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211, projection=ccrs.PlateCarree())
    c2 = ax.pcolormesh(
        model_data.lon, model_data.lat, xr.where(model_data.islnd.values == 1, model_data.pr, np.nan),
        vmin=0, vmax=300, cmap='Reds')
    cm = plt.get_cmap()
    cm.set_bad(alpha=0)
    cax2 = fig.add_axes([0.84, 0.55, 0.012, 0.3])
    cb2 = fig.colorbar(c2, cax=cax2, spacing='proportional', extend='max', label='rain rate / mm day$^{-1}$')
    # c2 = ax.contour(ivt_data.lon-180, ivt_data.lat, ivt_data.ivt.isel(time=0), cmap='Reds', alpha=0.5)
    c3 = ax.contour(
        model_data.lon, model_data.lat, model_data.shape, 
        levels=np.unique(model_data.shape.values)[:-1], colors='green', alpha=1, linewidths=0.8)

    ivt_abs_model = np.sqrt(model_data.ivtx**2 + model_data.ivty**2)
    ivt_abs_era5 = np.sqrt(era5_data.ivtx**2 + era5_data.ivty**2)

    c4 = ax.quiver(
        model_data.lon[::3], model_data.lat[::3], 
        model_data.ivtx.where(np.logical_and(ivt_abs_model>100, model_data.ivtx>0))[::3, ::3], 
        model_data.ivty.where(np.logical_and(ivt_abs_model>100, model_data.ivtx>0))[::3, ::3], 
        animated=True, scale=1500, scale_units='inches', width=0.002, alpha=0.8)
    qk = ax.quiverkey(c4, 0.8, 1.05, 100, r'100 $\mathrm{kg\,m\,s^{-1}}$', labelpos='E')
    ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree())
    c3 = ax2.pcolormesh(
        ivt_abs_era5.lon, ivt_abs_era5.lat, 
        (ivt_abs_model.where(np.logical_and(ivt_abs_model>100, model_data.ivtx>0)).values-
         ivt_abs_era5.where(np.logical_and(ivt_abs_era5>100, era5_data.ivtx>0)).values),
        #  /ivt_abs_era5.where(np.logical_and(ivt_abs_era5>100, ivtx>0)).values, 
        vmin=-500, vmax=500, cmap='coolwarm')
    cax3 = fig.add_axes([0.84, 0.14, 0.012, 0.3])
    cb3 = fig.colorbar(c3, cax=cax3, spacing='proportional', extend='max', label=r'$\Delta$ IVT / $\mathrm{kg\,m\,s^{-1}}$') 
    #$\frac{IVT_{AM4}-IVT_{ERA5}}{IVT_{ERA5}}$ / -
    c3 = ax2.contour(
        era5_data.lon, era5_data.lat, era5_data.shape, 
        levels=np.unique(era5_data.shape.values)[:-1], colors='green', alpha=1, linewidths=0.8)
    c3 = ax2.contour(
        era5_data.lon, era5_data.lat, era5_data.shape, 
        levels=np.unique(era5_data.shape.values)[:-1], colors='green', alpha=1, linewidths=0.8)
    c4 = ax2.quiver(
        era5_data.lon[::3], era5_data.lat[::3], 
        era5_data.ivtx.where(np.logical_and(ivt_abs_era5>100, era5_data.ivtx>0))[::3, ::3], 
        era5_data.ivty.where(np.logical_and(ivt_abs_era5>100, era5_data.ivtx>0))[::3, ::3], 
        animated=True, scale=1500, scale_units='inches', width=0.002, alpha=0.8)
    
    for axis in [ax, ax2]:
        axis.set_extent([-170, -80, 10, 60], crs=ccrs.PlateCarree())
        axis.coastlines("10m")
        # axis.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='tab:blue', edgecolor='tab:blue', linewidth=0.5)
        # axis.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='tab:blue', linewidth=0.5)
        axis.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5)
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        axis.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.set(title='AM4')
    ax2.set(title='ERA5')

    return fig, ax

def _main():
    base_path = '/Users/mp1506/ftp2.gfdl.noaa.gov/pub/Ming.Zhao/'
    model_data = load_model_data(base_path, '2016', ['prw', 'pr', 'ivtx', 'ivty'], ar_analysis=True)
    era5_data = load_era5_data(base_path, '2016', ['ivtx', 'ivty'], ar_analysis=True)
    model_data = sel_na_westcoast(model_data)
    era5_data = sel_na_westcoast(era5_data)
    fig, ax = plot_ivt_bias_rr(model_data.sel(time='2016-01-15'), era5_data.sel(time='2016-01-15'))
    fig.suptitle(f'2016-01-15', x=0.4, y=0.9)
    plt.savefig('/Users/mp1506/AR_california_2016_analysis/plots/refactor_test_plot.png')


if __name__ == '__main__':
        _main()