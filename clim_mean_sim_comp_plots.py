
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import os
from glob import glob
import pandas as pd
from pathlib import Path
from global_land_mask import globe
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

import data_util as du


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

label_color_map = {
    'nudge_30min_ctrl': sns.color_palette('colorblind')[0],
    'HX_ctrl': sns.color_palette('colorblind')[1],
    'mswep': sns.color_palette('colorblind')[2],
    'gpm': sns.color_palette('colorblind')[7],
    'stageiv': sns.color_palette('colorblind')[4],
    'nudge_30min_p2K': sns.color_palette('colorblind')[5],
    'HX_p2K': sns.color_palette('colorblind')[6],
    'imerg': sns.color_palette('colorblind')[3]
}

def plot_clim_mean_wind_diff_maps(
        ref_data, dist_data, ref_data_label, dist_data_label, region='global',
        level='ref'):
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(231, projection=ccrs.PlateCarree())
    if level == '250':
        cmap_range = np.arange(0, 31.5, 1.5)
        cmap_diff_range = np.arange(-4, 4.4, 0.4)
    elif level == '700':
        cmap_range = np.arange(0, 15.75, 0.75)
        cmap_diff_range = np.arange(-1.5, 1.65, 0.15)
    elif level == 'ref':
        cmap_range = np.arange(0, 10.5, 0.5)
        cmap_diff_range = np.arange(-1.5, 1.65, 0.15)
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data[f'windspeed_{level}'], 
        levels=cmap_range, cmap='viridis', extend='both')
    if region == 'global':
        dq = 10
    elif region == 'NA':
        dq = 5
    qu1 = ax1.quiver(
        ref_data.lon[::dq], ref_data.lat[::dq], 
        ref_data[f'u_{level}'][::dq, ::dq], ref_data[f'v_{level}'][::dq, ::dq],
        )
    if level == 'ref':
        label = f'10 m'
    else:
        label = f'{level} hPa'
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} {label} winds '+ '/ m s$^{-1}$', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(232, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data[f'windspeed_{level}'], 
        levels=cmap_range, cmap='viridis', extend='both')
    qu2 = ax2.quiver(
        dist_data.lon[::dq], dist_data.lat[::dq], 
        dist_data[f'u_{level}'][::dq, ::dq], dist_data[f'v_{level}'][::dq, ::dq],
        )
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} {label} winds '+ '/ m s$^{-1}$', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    unit = 'm s$^{-1}$'
    ax3 = fig.add_subplot(234, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data[f'windspeed_{level}'] - ref_data[f'windspeed_{level}'])/norm, 
        levels=cmap_diff_range, cmap='coolwarm', extend='both')
    qu3 = ax3.quiver(
        ref_data.lon[::5], ref_data.lat[::5], 
        dist_data[f'u_{level}'][::5, ::5] - ref_data[f'u_{level}'][::5, ::5], 
        dist_data[f'v_{level}'][::5, ::5] - ref_data[f'v_{level}'][::5, ::5],
        )
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} {label} winds / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = ref_data[f'windspeed_{level}'] / 100
    cmap_diff_range = np.arange(-50, 55, 5)
    unit = '%'
    ax4 = fig.add_subplot(235, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data[f'windspeed_{level}'] - ref_data[f'windspeed_{level}'])/norm, 
        levels=cmap_diff_range, cmap='coolwarm', extend='both')
    qu4 = ax4.quiver(
        ref_data.lon[::5], ref_data.lat[::5], 
        dist_data[f'u_{level}'][::5, ::5] - ref_data[f'u_{level}'][::5, ::5], 
        dist_data[f'v_{level}'][::5, ::5] - ref_data[f'v_{level}'][::5, ::5],
        )
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} {label} winds / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    unit = 'm s$^{-1}$'
    ax5 = fig.add_subplot(233)
    ax5.plot(
        ref_data[f'windspeed_{level}'].mean('lon'), 
        ref_data.lat,
        color='black',
        label=ref_data_label
    )
    ax5.plot(
        dist_data[f'windspeed_{level}'].mean('lon'), 
        dist_data.lat,
        color='firebrick',
        label=dist_data_label
    )
    ax5.set(
        xlabel=f'{label} zonal mean winds / {unit}',
        ylabel='Latitude')
    ax5.legend()
    ax6 = fig.add_subplot(236) 
    ax6.plot(
        dist_data[f'windspeed_{level}'].mean('lon') - ref_data[f'windspeed_{level}'].mean('lon'), 
        dist_data.lat,
        color='firebrick',
        label=f'{dist_data_label}-{ref_data_label}'
    )
    ax6.set(
        xlabel=f'$\Delta${label} zonal mean winds / {unit}',
        ylabel='Latitude')
    ax6.legend()
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    return fig, axs

def plot_clim_mean_pr_frequency_diff_map(
    ref_data, dist_data, ref_data_label, dist_data_label, 
    start_year, end_year, var='pr_frequency', months=np.arange(13)):
    all_days = np.arange(f"{start_year}-01-01", f"{end_year+1}-01-01", 
                           dtype='datetime64[D]')
    days_in_months = all_days[np.isin([date.month for date in all_days.astype(object)], months)]
    n_days = len(days_in_months)
    
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    levels = np.linspace(0, 80, 20)
    cmap = 'viridis'
    norm = n_days/100
    unit = '%'
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        (ref_data[var]/norm).values, 
        levels=levels, cmap=cmap, extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label}\n{var} / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    cb1.ax.set_xticks(
        np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        (dist_data[var]/norm).values, 
        levels=levels, cmap=cmap, extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label}\n{var} / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    cb2.ax.set_xticks(
        np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
    # Abs. diff
    norm = n_days / 100
    levels = np.arange(-10, 11, 1)
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, dist_data.lat, 
        ((dist_data[var].values - ref_data[var].values)/norm), 
        levels=levels, cmap='coolwarm', extend='both')
    c3 = ax3.contour(
        ref_data.lon, dist_data.lat, 
        (ref_data[var]/n_days).values, 
        levels=5, colors='gray', linewidths=0.7)
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label}\n{var}' + ' / %', 
        orientation='horizontal', ax=ax3, pad=0.05, shrink=0.6)
    cb3.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], 
                                (levels[-1]-levels[0])/5))
    # Rel. diff
    norm = ref_data[var] / 100
    levels = np.arange(-50, 56, 5)
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, dist_data.lat, 
        ((dist_data[var].values - ref_data[var].values)/norm).values, 
        levels=levels, cmap='coolwarm', extend='both')
    c4 = ax4.contour(
        ref_data.lon, dist_data.lat, 
        (ref_data[var]/n_days).values, 
        levels=5, colors='gray', linewidths=0.7)
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label}/{ref_data_label}\n{var}' + ' / %', 
        orientation='horizontal', ax=ax4, pad=0.05, shrink=0.6)
    cb4.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], 
                                (levels[-1]-levels[0])/5))
    dist_data['lat'] = ref_data['lat']
    dist_data['lon'] = ref_data['lon']
    bias = (dist_data[var].mean()/n_days - ref_data[var].mean()/n_days).values*100
    rmse = root_mean_squared_error(
        ref_data[var].stack(point=('lat', 'lon')).dropna('point').values/n_days, 
        dist_data[var].stack(point=('lat', 'lon')).dropna('point').values/n_days)
    ax3.set_title(
        f'bias={np.round(bias, 2)} % \t'
        f'rmse={np.round(rmse, 2)} %')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_precip_diff_maps(
        ref_data, dist_data, ref_data_label, dist_data_label, region='global', var='precip'):
    if var in ['pr_intensity', 'prsn_intensity']:
        mean_levels = np.arange(0, 16, 1)
        abs_diff_levels = np.arange(-2, 2.1, 0.1)
        rel_diff_levels = np.arange(-50, 55, 5)
    else:
        mean_levels = np.arange(0, 10.5, 0.5)
        abs_diff_levels = np.arange(-1.5, 1.6, 0.1)
        rel_diff_levels = np.arange(-50, 55, 5)
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data[var].values*86400,
        levels=mean_levels, cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} {var} '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data[var].values*86400, 
        levels=mean_levels, cmap='viridis', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} {var} '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    unit = 'mm day$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data[var].values - ref_data[var].values)*86400/norm, 
        levels=abs_diff_levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} {var} / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = ref_data[var].values / 100
    unit = '%'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data[var].values - ref_data[var].values)/norm, 
        levels=rel_diff_levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} {var} / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{np.round(ref_data[var].mean().values*86400, 2)} mm day$^{{{-1}}}$')
    ax2.set_title(f'{np.round(dist_data[var].mean().values*86400, 2)} mm day$^{{{-1}}}$')
    dist_data['lat'] = ref_data['lat']
    dist_data['lon'] = ref_data['lon']
    bias = (dist_data[var].mean() - ref_data[var].mean()).values*86400
    rmse = root_mean_squared_error(
        ref_data[var].stack(point=('lat', 'lon')).dropna('point').values, 
        dist_data[var].stack(point=('lat', 'lon')).dropna('point').values)*86400
    ax3.set_title(
        f'bias={np.round(bias, 2)} mm day$^{{{-1}}}$\t'
        f'rmse={np.round(rmse, 2)} mm day$^{{{-1}}}$')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_olr_diff_maps(ref_data, dist_data, ref_data_label, dist_data_label, region='global'):
    unit = 'W m$^{-2}$'
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.olr, 
        levels=np.arange(0, 310, 10), cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} olr / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.olr, 
        levels=np.arange(0, 310, 10), cmap='viridis', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} olr / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    levels = np.arange(-20, 22, 2)
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.olr - ref_data.olr)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} olr / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = ref_data.olr / 100
    levels = np.arange(-10, 11, 1)
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.olr - ref_data.olr)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} olr / %', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{str(np.round(ref_data.olr.mean().values, 2))} {unit}')
    ax2.set_title(f'{str(np.round(dist_data.olr.mean().values, 2))} {unit}')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_evap_diff_maps(ref_data, dist_data, ref_data_label, dist_data_label, region='global'):
    unit = 'kg m$^{-2}$ day$^{-1}$'
    scaling = 86400
    levels = np.arange(-2, 9, 1)
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.evap*scaling, 
        levels=levels, cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} evap / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.evap*scaling, 
        levels=levels, cmap='viridis', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} evap / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    levels = np.arange(-1, 1, 0.1)
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.evap - ref_data.evap)*scaling/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} evap / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6, format=tkr.FormatStrFormatter('%.1f'))
    # cb3.ax.set_xticklabels([round(l, 1) for l in levels])
    norm = ref_data.evap / 100
    levels = np.arange(-30, 33, 3)
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.evap - ref_data.evap)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} evap / %', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{str(np.round(ref_data.evap.mean().values*scaling, 2))} {unit}')
    ax2.set_title(f'{str(np.round(dist_data.evap.mean().values*scaling, 2))} {unit}')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_rv_d_h2o_diff_maps(ref_data, dist_data, ref_data_label, dist_data_label, region='global'):
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.rv_d_h2o*86400, 
        levels=np.arange(0, 210, 10), cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} rv_d_h2o '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.rv_d_h2o*86400, 
        levels=np.arange(0, 210, 10), cmap='viridis', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} rv_d_h2o '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    levels = np.arange(-20, 22, 2)
    unit = 'mm day$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.rv_d_h2o - ref_data.rv_d_h2o)*86400/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} rv_d_h2o / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = ref_data.rv_d_h2o*86400 / 100
    levels = np.arange(-10, 11, 1)
    unit = '%'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.rv_d_h2o - ref_data.rv_d_h2o)*86400/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} rv_d_h2o / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{str(np.round(ref_data.rv_d_h2o.sum().values*86400, 2))} mm day$^{{{-1}}}$')
    ax2.set_title(f'{str(np.round(dist_data.rv_d_h2o.sum().values*86400, 2))} mm day$^{{{-1}}}$')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs


def plot_clim_mean_diff_maps(
    ref_data, dist_data, ref_data_label, dist_data_label, variable, 
    start_year, end_year, region='global', months=np.arange(13)): 
    if region == 'NA':
        ref_data = du.sel_na(ref_data)
        dist_data = du.sel_na(dist_data)
    elif region == 'conus_land':
        ref_data = du.sel_conus_land(ref_data)
        dist_data = du.sel_conus_land(dist_data)

    if variable == 'ref_winds':
        ref_data['windspeed_ref'] = np.sqrt(ref_data.u_ref**2 + ref_data.v_ref**2)
        dist_data['windspeed_ref'] = np.sqrt(dist_data.u_ref**2 + dist_data.v_ref**2)
        fig, axs = plot_clim_mean_wind_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region, level='ref')
    if variable == '700_winds':
        ref_data['windspeed_700'] = np.sqrt(ref_data.u_700**2 + ref_data.v_700**2)
        dist_data['windspeed_700'] = np.sqrt(dist_data.u_700**2 + dist_data.v_700**2)
        fig, axs = plot_clim_mean_wind_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region, level='700')
    if variable == '250_winds':
        ref_data['windspeed_250'] = np.sqrt(ref_data.u_250**2 + ref_data.v_250**2)
        dist_data['windspeed_250'] = np.sqrt(dist_data.u_250**2 + dist_data.v_250**2)
        fig, axs = plot_clim_mean_wind_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region, level='250')
    if variable in ['precip', 'pr_intensity']:
        fig, axs = plot_clim_mean_precip_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region, variable)
    if variable in ['pr_frequency']:
        fig, axs = plot_clim_mean_pr_frequency_diff_map(
            ref_data, dist_data, ref_data_label, dist_data_label, 
            start_year, end_year, variable, months)
    if variable == 'olr':
        fig, axs = plot_clim_mean_olr_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region)
    if variable == 'evap':
        fig, axs = plot_clim_mean_evap_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region)
    if variable == 'rv_d_h2o':
        fig, axs = plot_clim_mean_rv_d_h2o_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region)
    [ax.coastlines("50m", linewidth=0.5) for ax in axs 
     if str(ax.__class__) == "<class 'cartopy.mpl.geoaxes.GeoAxes'>"]
    [ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5) for ax in axs
     if str(ax.__class__) == "<class 'cartopy.mpl.geoaxes.GeoAxes'>"]
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
    [ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5) for ax in axs
     if str(ax.__class__) == "<class 'cartopy.mpl.geoaxes.GeoAxes'>"]
    plt.tight_layout()
    return fig, axs

def is_winter(month):
    return (month >= 11) | (month <= 2)

def plot_clim_mean_sim_comp(
    exp_name_ctrl, exp_name_dist, 
    ctrl_label, dist_label, ref_sub, start_year, end_year,
    variables, months=range(13), time_str='all_months', regions=['global', 'NA']):
    unique_vars = np.unique(np.concatenate([var for var in variables.values()]))
    ts_all_path = {var: (exp_basepath_map[exp_name_ctrl]
                            if (var not in ['u_700', 'v_700', 'u_250', 'v_250', 
                                            'pr_intensity', 'prsn_intensity', 'pr_frequency'])
                            and (exp_name_ctrl != 'c192_obs')
                            else '/archive/Marc.Prange/ts_all_missing_vars/')
                    for var in np.concatenate([var for var in variables.values()])              
                    }
    if exp_name_ctrl != 'c192_obs':
        ref_paths = [glob(f'{ts_all_path[v]}{exp_name_ctrl}/ts_all/{sub}*.{v}.nc')[0]
                        for sub in variables.keys() 
                        for v in variables[sub]]
    else:
        ref_paths = [glob(f'{ts_all_path[v]}{exp_name_ctrl}/ts_all/{ref_sub}*.{v}.nc')[0]
                     for v in variables[ref_sub]]
    ts_all_path = {var: (exp_basepath_map[exp_name_dist]
                            if var not in ['u_700', 'v_700', 'u_250', 'v_250', 
                                           'pr_intensity', 'prsn_intensity', 'pr_frequency']
                            and (exp_name_dist != 'c192_obs')
                            else '/archive/Marc.Prange/ts_all_missing_vars/')
                    for var in np.concatenate([var for var in variables.values()])              
                    }
    dist_paths = [glob(f'{ts_all_path[v]}{exp_name_dist}/ts_all/{sub}*.{v}.nc')[0] 
                    for sub in variables.keys()
                    for v in variables[sub] if sub != ref_sub]
    print(f'Loading clim. mean data for simulation {ctrl_label}', flush=True)
    ref_data = du.lon_360_to_180(
            xr.open_mfdataset(ref_paths, compat='override').sel(
        time=slice(f'{start_year}', f'{end_year}')))
    ref_data = ref_data.sel(time=ref_data.time.dt.month.isin(months))
    
    ref_data_mean = ref_data.mean('time')
    print(f'Loading clim. mean data for simulation {dist_label}', flush=True)
    dist_data = du.lon_360_to_180(
            xr.open_mfdataset(dist_paths, compat='override').sel(
        time=slice(f'{start_year}', f'{end_year}')))
    dist_data = dist_data.sel(time=dist_data.time.dt.month.isin(months))
    dist_data_mean = dist_data.mean('time')
    if 'pr_frequency' in unique_vars:
        ref_data_mean['pr_frequency'] = ref_data.pr_frequency.sum('time')
        dist_data_mean['pr_frequency'] = dist_data.pr_frequency.sum('time')
    
    plot_configs = {
    'region': regions,
    }
    for region in plot_configs['region']:
        for var in unique_vars:
            print(
                f'Plotting {var} for simulations {dist_label}-{ctrl_label} over {region}.', 
                flush=True)
            fig, axs = plot_clim_mean_diff_maps(
                ref_data_mean, dist_data_mean, ctrl_label, dist_label, var, 
                start_year, end_year, region, months)
            fig_dir = f'plots/clim_mean_sim_comp/{dist_label}-{ctrl_label}/'
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                        fig_dir +
                        f'{ctrl_label}_{dist_label}_{start_year}-{end_year}_clim_mean_{var}_{region}_{time_str}.png',
                        dpi=300, bbox_inches='tight')

def plot_daily_cdf_for_region(
        exp_names, exp_labels, start_year, end_year, variables, months, time_str, min_vals):
    fig = plt.figure(figsize=(len(variables)*5, 5))
    for ivar, (var, min_val) in enumerate(zip(variables, min_vals)):
        ax = fig.add_subplot(1, len(variables), ivar+1)
        for exp_name, exp_label in zip(exp_names, exp_labels):
            if isinstance(exp_name, dict):
                exp_name_subs = list(exp_name.values())[0]
                exp_name = list(exp_name.keys())[0]
                for exp_name_sub in exp_name_subs:
                    print(f"Loading {exp_name_sub}.{exp_label} data", flush=True)
                    if var == 'ar_pr':
                        obs_var = 'ar_precip'
                    elif var == 'pr':
                        obs_var = 'precip'
                    else:
                        obs_var = var
                    data = xr.open_mfdataset(
                        f'/archive/Marc.Prange/na_data/{exp_name}/{exp_name_sub}.{exp_name}_na_*.nc'
                    )[obs_var].sel(time=slice(str(start_year), str(end_year)))
                    data = data.sel(time=data.time.dt.month.isin(months))
                    data = du.sel_conus_land(data)
                    data = data.where(data >= min_val)
                    data = data.stack(case=('lat', 'lon', 'time')).dropna('case')
                    pr_x = np.logspace(-3, np.log10(90), 200, endpoint=True)
                    pctls = (100 - pr_x)
                    f1=interp1d(pctls, pr_x, fill_value='extrapolate')
                    pctl_values = np.percentile(data.values, pctls)
                    ax.plot(pr_x, pctl_values*86400, label=exp_name_sub, linewidth=1, color=label_color_map[exp_name_sub])
            else:
                print(f"Loading {exp_label} data", flush=True)
                data = xr.open_mfdataset(
                    f'/archive/Marc.Prange/na_data/{exp_name}/{exp_name}_na_*.nc'
                )[var].sel(time=slice(str(start_year), str(end_year)))
                data = data.sel(time=data.time.dt.month.isin(months))
                data = du.sel_conus_land(data)
                data = data.where(data >= min_val)
                data = data.stack(case=('lat', 'lon', 'time')).dropna('case')
                pr_x = np.logspace(-3, np.log10(90), 200, endpoint=True)
                pctls = (100 - pr_x)
                f1=interp1d(pctls, pr_x, fill_value='extrapolate')
                # pctls = np.logspace(1, np.log10(99.999), num=100, base=10, endpoint=True)
                # pctls = np.linspace(10, 99.999, num=1000, endpoint=True)
                pctl_values = np.percentile(data.values, pctls)
                ax.plot(pr_x, pctl_values*86400, label=exp_label, linewidth=1, color=label_color_map[exp_label])
        ax.legend()
        ax.set(
            ylabel=f'{var}'+' / mm day$^{-1}$', 
            xlabel='Percentile', 
            ylim=[1e-1, 500], 
            # ylim=[0, 100]
            )
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xticks((f1([10,90,99,99.9,99.99,99.999])))
        ax.set_xticklabels(['10','90','99','99.9','99.99','99.999'])
        ax.set_xlim(0.001,90) 
        ax.invert_xaxis()
    plt.tight_layout()
    print("Storing figure...", flush=True)
    s = '_'
    plt.savefig(
         'plots/clim_mean_sim_comp/histograms/pr_percentile_dist_'
        f'{s.join(variables)}_{s.join(exp_labels)}_{start_year}_{end_year}_{time_str}_conus_loglog_reversed.png',
        dpi=300
    )

def plot_daily_cdf_diff_to_obs(
        exp_name, exp_label, obs_names, start_year, end_year, variable, min_val, months, time_str):
    pr_x = np.logspace(-3, np.log10(90), 200, endpoint=True)
    pctls = (100 - pr_x)
    f1=interp1d(pctls, pr_x, fill_value='extrapolate')
    fig = plt.figure(figsize=(5, 6))
    gs = gridspec.GridSpec(
        nrows=2, ncols=1, height_ratios=[1, 2], hspace=0.12)
    ax2 = fig.add_subplot(gs[1])
    ax1 = fig.add_subplot(gs[0], sharex=ax2)
    exp_data = xr.open_mfdataset(
                    f'/archive/Marc.Prange/na_data/{exp_name}/{exp_name}_na_*.nc'
                )[variable].sel(time=slice(str(start_year), str(end_year)))
    exp_data = exp_data.sel(time=exp_data.time.dt.month.isin(months))
    exp_data = du.sel_conus_land(exp_data)
    exp_data = exp_data.stack(case=('lat', 'lon', 'time')).dropna('case')
    exp_data_sum = exp_data.sum().values
    exp_data = exp_data.where(exp_data >= min_val).dropna('case')
    exp_pctl_values = np.percentile(exp_data.values, pctls)
    exp_contr_to_total = np.array(
        [exp_data.where(exp_data < pctl_val).sum().values / exp_data_sum
         for pctl_val in exp_pctl_values])
    exp_contr_to_total_count = np.array(
        [(exp_data < pctl_val).sum() / len(exp_data)
         for pctl_val in exp_pctl_values])
    ax1.plot(exp_pctl_values*86400, exp_contr_to_total*100, 
             label=f'{variable} sum', lw=1, color=label_color_map[exp_label])
    ax1.plot(exp_pctl_values*86400, exp_contr_to_total_count*100, 
             label=f'{variable} count', lw=1, ls='--', color=label_color_map[exp_label])
    ax1.legend()
    for obs_name in obs_names:
        obs_data = xr.open_mfdataset(
                    f'/archive/Marc.Prange/na_data/c192_obs/{obs_name}.c192_obs_na_*.nc'
                )[variable].sel(time=slice(str(start_year), str(end_year)))
        obs_data = obs_data.sel(time=obs_data.time.dt.month.isin(months))
        obs_data = du.sel_conus_land(obs_data)
        obs_data = obs_data.stack(case=('lat', 'lon', 'time')).dropna('case')
        obs_data_sum = obs_data.sum().values
        obs_data = obs_data.where(obs_data >= min_val).dropna('case')
        obs_pctl_values = np.percentile(obs_data.values, pctls)
        obs_contr_to_total = np.array(
            [obs_data.where(obs_data < pctl_val).sum().values / obs_data_sum
         for pctl_val in obs_pctl_values])
        obs_contr_to_total_count = np.array(
        [(obs_data < pctl_val).sum() / len(obs_data)
         for pctl_val in obs_pctl_values])
        ax1.plot(exp_pctl_values*86400, obs_contr_to_total*100, 
                 lw=1, color=label_color_map[obs_name])
        ax1.plot(exp_pctl_values*86400, obs_contr_to_total_count*100, 
                 lw=1, ls='--', color=label_color_map[obs_name])
        ax2.plot(
            exp_pctl_values*86400, (exp_pctl_values-obs_pctl_values)/obs_pctl_values*100, 
            label=f'{exp_label}-{obs_name}', lw=1, color=label_color_map[obs_name])
    ax1.set(
        ylabel=f'cumsum/sum {variable} / %',
        ylim=[1e-2, 1e2],
    )
    ax2.hlines(0, 1e-1, 250, color='gray', ls='--')
    ax2.legend()
    ax2.set(
        xlabel=f'{variable}'+' / mm day$^{-1}$', 
        ylabel='$\Delta_{rel}$'+f'{exp_label}-obs / %', 
        xlim=[1e-1, 250], 
        ylim=[-100, 100]
    )
    ax2.set_xscale('log')
    # plt.tight_layout()
    print("Storing figure...", flush=True)
    s = '_'
    plt.savefig(
         'plots/clim_mean_sim_comp/histograms/pctl_rel_obs_diff_'
        f'{variable}_{exp_label}_{s.join(obs_names)}_{start_year}_{end_year}_{time_str}_conus.png',
        dpi=300, bbox_inches='tight'
    )

def plot_daily_cdf_change_with_warming(
        exp_name_ref, exp_label_ref, exp_name_dist, exp_label_dist, 
        start_year, end_year, variables, min_vals, months, time_str):
    dT = du.get_global_mean_dT(exp_name_ref, exp_name_dist, start_year, end_year)
    pr_x = np.logspace(-3, np.log10(90), 200, endpoint=True)
    pctls = (100 - pr_x)
    f1=interp1d(pctls, pr_x, fill_value='extrapolate')
    fig = plt.figure(figsize=(5, 6))
    gs = gridspec.GridSpec(
        nrows=1, ncols=1, height_ratios=[1], hspace=0.12)
    ax2 = fig.add_subplot(gs[0])
    for variable, min_val in zip(variables, min_vals):
        exp_data_ref = xr.open_mfdataset(
                        f'/archive/Marc.Prange/na_data/{exp_name_ref}/{exp_name_ref}_na_*.nc'
                    )[variable].sel(time=slice(str(start_year), str(end_year)))
        exp_data_ref = exp_data_ref.sel(time=exp_data_ref.time.dt.month.isin(months))
        exp_data_ref = du.sel_conus_land(exp_data_ref)
        exp_data_ref = exp_data_ref.stack(case=('lat', 'lon', 'time'))
        exp_data_ref = exp_data_ref.where(exp_data_ref >= min_val).dropna('case')
        exp_ref_pctl_values = np.percentile(exp_data_ref.values, pctls)

        exp_data_dist = xr.open_mfdataset(
                        f'/archive/Marc.Prange/na_data/{exp_name_dist}/{exp_name_dist}_na_*.nc'
                    )[variable].sel(time=slice(str(start_year), str(end_year)))
        exp_data_dist = exp_data_dist.sel(time=exp_data_dist.time.dt.month.isin(months))
        exp_data_dist = du.sel_conus_land(exp_data_dist)
        exp_data_dist = exp_data_dist.stack(case=('lat', 'lon', 'time'))
        exp_data_dist = exp_data_dist.where(exp_data_dist >= min_val).dropna('case')
        exp_dist_pctl_values = np.percentile(exp_data_dist.values, pctls)
        ax2.plot(
            exp_ref_pctl_values*86400, (exp_dist_pctl_values-exp_ref_pctl_values)/exp_ref_pctl_values/dT*100, 
            label=f'{variable}', lw=1)
    ax2.hlines(0, 1e-1, 250, color='gray', ls='--')
    ax2.legend()
    ax2.set(
        xlabel=f'{variable}'+' / mm day$^{-1}$', 
        ylabel='$\Delta_{rel}$'+f'{exp_label_dist}-{exp_label_ref} /'+' % K$^{-1}$', 
        xlim=[1e-1, 250], 
        ylim=[-2, 10]
    )
    ax2.set_xscale('log')
    # plt.tight_layout()
    print("Storing figure...", flush=True)
    plt.savefig(
         'plots/clim_mean_sim_comp/histograms/pctl_rel_warming_diff_'
        f'{variable}_{exp_label_dist}-{exp_label_ref}_{start_year}_{end_year}_{time_str}_conus.png',
        dpi=300, bbox_inches='tight'
    )


def _main():
    for exp_name_dist, dist_label in zip(['c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min'],
                                          ['nudge_30min_ctrl']):
        for ctrl_label in ['imerg', 'stageiv', 'mswep']:
            exp_name_ctrl = 'c192_obs'
            # exp_name_dist = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min'
            # ctrl_label = 'stageiv'
            # dist_label = 'nudge_30min_ctrl'
            start_year = 2001
            end_year = 2020
            variables = {
                'atmos_cmip': ['ar_precip', 'ar_pr_intensity', 'ar_pr_frequency'], 
                'atmos': [], 
                'land': [], 
                'river': [], 
                ctrl_label: ['ar_precip', 'ar_pr_intensity', 'ar_pr_frequency']}
            ref_sub = ctrl_label
            for months, time_str in zip([[11, 12, 1, 2], np.arange(1, 13)], ['winter', 'all_months']):
                plot_clim_mean_sim_comp(
                    exp_name_ctrl, exp_name_dist, 
                    ctrl_label, dist_label, ref_sub, start_year, end_year,
                    variables, months, time_str=time_str, regions=['conus_land'])
    exp_names = [
        # 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min',
        # 'c192L33_am4p0_amip_HIRESMIP_HX_p2K',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min',
        # 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K', 
        {'c192_obs': ['mswep', 'imerg', 'stageiv']},
    ]
    exp_labels = [
        # 'HX_ctrl',
        # 'HX_p2K',
        'nudge_30min_ctrl',
        # 'nudge_30min_p2K',
        'c192_obs'
    ]

    variables = ['ar_pr', 'pr']
    start_year = 2001
    end_year = 2020
    months = np.arange(1, 13)
    time_str = 'all_months'
    plot_daily_cdf_for_region(
        exp_names, exp_labels, start_year, end_year, variables, months, time_str, min_vals=[0, 0])
    # plot_daily_cdf_diff_to_obs(
    #     exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min', 
    #     exp_label='nudge_30min_ctrl',
    #     obs_names=['mswep', 'stageiv', 'imerg'],
    #     start_year=start_year, end_year=end_year, 
    #     variable=variables[0], min_val=1/86400, months=months, time_str=time_str)
    
    # plot_daily_cdf_change_with_warming(
    #     exp_name_ref='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min', 
    #     exp_label_ref='nudge_30min_ctrl',
    #     exp_name_dist='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K', 
    #     exp_label_dist='nudge_30min_p2K',
    #     start_year=1951, end_year=2020, 
    #     variables=['ar_pr', 'pr'], min_vals=[1/86400, 1/86400], months=months, time_str=time_str)

if __name__ == '__main__':
    _main()
