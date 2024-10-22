
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

def plot_clim_mean_wind_diff_maps(
        ref_data, dist_data, ref_data_label, dist_data_label, dT, level='ref'):
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(231, projection=ccrs.PlateCarree())
    if level == '250':
        cmap_range = np.arange(0, 31.5, 1.5)
        cmap_diff_range = np.arange(-1, 1.1, 0.1)
    elif level == '700':
        cmap_range = np.arange(0, 15.75, 0.75)
        cmap_diff_range = np.arange(-0.5, 0.6, 0.1)
    elif level == 'ref':
        cmap_range = np.arange(0, 10.5, 0.5)
        cmap_diff_range = np.arange(-0.5, 0.6, 0.1) 
    cmap = 'viridis'
    norm = 1
    unit = 'm s$^{-1}$'
    if dist_data_label == 'delta_HX':
        if level == '250':
            cmap_range = np.arange(-1, 1.1, 0.1)
            cmap_diff_range = np.arange(-1, 1.1, 0.1)
        elif level == '700':
            cmap_range = np.arange(-0.5, 0.6, 0.1)
            cmap_diff_range = np.arange(-0.5, 0.6, 0.1)
        elif level == 'ref':
            cmap_range = np.arange(-0.5, 0.6, 0.1)
            cmap_diff_range = np.arange(-0.5, 0.6, 0.1)
        cmap = 'coolwarm' 
        norm = dT
        unit = 'm s$^{-1}$ K$^{-1}$'
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data[f'windspeed_{level}']/norm, 
        levels=cmap_range, cmap=cmap, extend='both')
    dq = int(len(ref_data.lon)/80)

    qu1 = ax1.quiver(
        ref_data.lon[::dq], ref_data.lat[::dq], 
        ref_data[f'u_{level}'][::dq, ::dq], ref_data[f'v_{level}'][::dq, ::dq],
        )
    if level == 'ref':
        label = f'10 m'
    else:
        label = f'{level} hPa'
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} {label} winds / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(232, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data[f'windspeed_{level}'], 
        levels=cmap_range, cmap=cmap, extend='both')
    qu2 = ax2.quiver(
        dist_data.lon[::dq], dist_data.lat[::dq], 
        dist_data[f'u_{level}'][::dq, ::dq], dist_data[f'v_{level}'][::dq, ::dq],
        )
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} {label} winds / {unit}', 
        orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = dT
    unit = 'm s$^{-1}$ K$^{-1}$'
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
    norm = dT*ref_data[f'windspeed_{level}'] / 100
    cmap_diff_range = np.arange(-25, 30, 5)
    unit = '% $K^{-1}$'
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
    norm = dT
    if dist_data_label == 'delta_HX':
        unit = 'm s$^{-1}$ K$^{-1}$'
        norm = 1
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
        (dist_data[f'windspeed_{level}'].mean('lon') - ref_data[f'windspeed_{level}'].mean('lon'))/norm, 
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

def plot_clim_mean_precip_diff_maps(
        ref_data, dist_data, ref_data_label, dist_data_label, dT, var='precip'):
    
    if var in ['pr_intensity', 'prsn_intensity']:
        mean_levels = np.arange(0, 16, 1)
        abs_diff_levels = np.arange(-1, 1.1, 0.1)
        rel_diff_levels = np.arange(-10, 11, 1)
    elif var in ['ar_pr_intensity']:
        mean_levels = np.arange(0, 21, 1)
        abs_diff_levels = np.arange(-1, 1.1, 0.1)
        rel_diff_levels = np.arange(-10, 11, 1)
    elif var in  ['precip']:
        mean_levels = np.arange(0, 10.5, 0.5)
        abs_diff_levels = np.arange(-0.5, 0.6, 0.1)
        rel_diff_levels = np.arange(-10, 11, 1)
    elif var  in ['ar_pr']:
        mean_levels = np.arange(0, 4.2, 0.2)
        abs_diff_levels = np.arange(-0.2, 0.22, 0.02)
        rel_diff_levels = np.arange(-10, 11, 1)
    cmap = 'viridis'
    norm = 86400**-1
    unit = 'mm day$^{-1}$'
    if dist_data_label == 'delta_HX':
        if var in ['pr_intensity', 'prsn_intensity', 'ar_pr_intensity']:
            mean_levels = np.arange(-1, 1.1, 0.1)
            abs_diff_levels = np.arange(-1, 1.1, 0.1)
            rel_diff_levels = np.arange(-10, 11, 1)
        elif var in ['precip', 'ar_pr']:
            mean_levels = np.arange(-0.5, 0.6, 0.1)
            abs_diff_levels = np.arange(-0.5, 0.6, 0.1)
            rel_diff_levels = np.arange(-10, 11, 1)
        cmap = 'coolwarm' 
        norm = dT/86400
        unit = 'mm day$^{-1}$ K$^{-1}$'
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data[var]/norm,
        levels=mean_levels, cmap=cmap, extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} {var} / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data[var]/norm, 
        levels=mean_levels, cmap=cmap, extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} {var} / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = dT/86400
    unit = 'mm day$^{-1}$ K$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data[var] - ref_data[var])/norm, 
        levels=abs_diff_levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} {var} / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = dT*ref_data[var] / 100
    unit = '% K$^{-1}$'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data[var] - ref_data[var])/norm, 
        levels=rel_diff_levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} {var} / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{np.round(ref_data[var].mean().values*86400, 2)} mm day$^{{{-1}}}$')
    ax2.set_title(f'{np.round(dist_data[var].mean().values*86400, 2)} mm day$^{{{-1}}}$')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_olr_diff_maps(
        ref_data, dist_data, ref_data_label, dist_data_label, dT):
    levels = np.arange(0, 310, 10)
    cmap = 'viridis'
    norm = 1
    unit = 'W m$^{-2}$'
    if dist_data_label == 'delta_HX':
        levels = np.arange(-10, 11, 1)
        cmap = 'coolwarm' 
        norm = dT
        unit = 'W m$^{-2}$ K$^{-1}$'

    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.olr/norm, 
        levels=levels, cmap=cmap, extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} olr / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.olr/norm, 
        levels=levels, cmap=cmap, extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} olr / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = dT
    levels = np.arange(-10, 11, 1)
    unit = 'W m$^{-2}$ K$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.olr - ref_data.olr)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} olr / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = dT*ref_data.olr / 100
    levels = np.arange(-5, 6, 1)
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.olr - ref_data.olr)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} olr '+'/ % K$^{-1}$', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{str(np.round(ref_data.olr.mean().values, 2))} {unit}')
    ax2.set_title(f'{str(np.round(dist_data.olr.mean().values, 2))} {unit}')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_pr_frequency_diff_map(
    ref_data, dist_data, ref_data_label, dist_data_label, dT, start_year, end_year, var='pr_frequency'):
    if var in ['pr_frequency']:
        mean_levels = np.linspace(0, 80, 20)
        abs_diff_levels = np.arange(-1, 1.1, 0.1)
        rel_diff_levels = np.arange(-10, 11, 1)
    elif var in ['ar_pr_frequency']:
        mean_levels = np.arange(0, 21, 1)
        abs_diff_levels = np.arange(-0.5, 0.6, 0.1)
        rel_diff_levels = np.arange(-5, 5.5, 0.5)
    n_days = len(np.arange(f"{start_year}-01-01", f"{end_year+1}-01-01", 
                           dtype='datetime64[D]'))
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cmap = 'viridis'
    norm = n_days/100
    unit = '%'
    if dist_data_label == 'delta_HX':
        cmap = 'coolwarm' 
        norm = n_days*dT/100
        unit = '% K$^{-1}$'
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data[var]/norm, 
        levels=mean_levels, cmap=cmap, extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label}\n{var} / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    cb1.ax.set_xticks(
        np.arange(mean_levels[0], mean_levels[-1]+np.diff(mean_levels)[-1], (mean_levels[-1]-mean_levels[0])/5))
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data[var]/norm, 
        levels=mean_levels, cmap=cmap, extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label}\n{var} / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    cb2.ax.set_xticks(
        np.arange(mean_levels[0], mean_levels[-1]+np.diff(mean_levels)[-1], (mean_levels[-1]-mean_levels[0])/5))
    # Abs. diff
    norm = dT * n_days / 100
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, dist_data.lat, 
        (dist_data[var] - ref_data[var])/norm, 
        levels=abs_diff_levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label}\n{var}' + ' / % K$^{-1}$', 
        orientation='horizontal', ax=ax3, pad=0.05, shrink=0.6)
    cb3.ax.set_xticks(np.arange(abs_diff_levels[0], abs_diff_levels[-1]+np.diff(abs_diff_levels)[-1], 
                                (abs_diff_levels[-1]-abs_diff_levels[0])/5))
    # Rel. diff
    norm = dT * ref_data[var] / 100
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, dist_data.lat, 
        (dist_data[var] - ref_data[var])/norm, 
        levels=rel_diff_levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label}/{ref_data_label}\n{var}' + ' / % K$^{-1}$', 
        orientation='horizontal', ax=ax4, pad=0.05, shrink=0.6)
    cb4.ax.set_xticks(np.arange(rel_diff_levels[0], rel_diff_levels[-1]+np.diff(rel_diff_levels)[-1], 
                                (rel_diff_levels[-1]-rel_diff_levels[0])/5))
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_ar_pr_to_pr_ratio_diff_map(
    ref_data, dist_data, ref_data_label, dist_data_label, dT, var='ar_pr_to_pr_ratio'):
    mean_levels = np.linspace(0, 50, 25)
    abs_diff_levels = np.arange(-5, 5.2, 0.2)
    rel_diff_levels = np.arange(-10, 11, 1)

    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cmap = 'viridis'
    unit = '%'
    norm = 1/100
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data[var]/norm, 
        levels=mean_levels, cmap=cmap, extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label}\n{var} / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    cb1.ax.set_xticks(
        np.arange(mean_levels[0], mean_levels[-1]+np.diff(mean_levels)[-1], (mean_levels[-1]-mean_levels[0])/5))
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data[var]/norm, 
        levels=mean_levels, cmap=cmap, extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label}\n{var} / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    cb2.ax.set_xticks(
        np.arange(mean_levels[0], mean_levels[-1]+np.diff(mean_levels)[-1], (mean_levels[-1]-mean_levels[0])/5))
    # Abs. diff
    norm = dT / 100
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, dist_data.lat, 
        (dist_data[var] - ref_data[var])/norm, 
        levels=abs_diff_levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label}\n{var}' + ' / % K$^{-1}$', 
        orientation='horizontal', ax=ax3, pad=0.05, shrink=0.6)
    cb3.ax.set_xticks(np.arange(abs_diff_levels[0], abs_diff_levels[-1]+np.diff(abs_diff_levels)[-1], 
                                (abs_diff_levels[-1]-abs_diff_levels[0])/5))
    # Rel. diff
    norm = dT * ref_data[var] / 100
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, dist_data.lat, 
        (dist_data[var] - ref_data[var])/norm, 
        levels=rel_diff_levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label}/{ref_data_label}\n{var}' + ' / % K$^{-1}$', 
        orientation='horizontal', ax=ax4, pad=0.05, shrink=0.6)
    cb4.ax.set_xticks(np.arange(rel_diff_levels[0], rel_diff_levels[-1]+np.diff(rel_diff_levels)[-1], 
                                (rel_diff_levels[-1]-rel_diff_levels[0])/5))
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_ar_pr_diff_to_pr_diff_ratio_map(
    ref_data, dist_data, ref_data_label, dist_data_label, dT):
    
    abs_diff_levels = np.arange(-0.15, 0.16, 0.01)
    rel_diff_levels = np.arange(-50, 55, 5)

    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(131, projection=ccrs.PlateCarree())
    cmap = 'coolwarm'
    unit = 'mm day$^{-1}$ K$^{-1}$'
    norm = dT/86400
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.ar_pr - ref_data.ar_pr)/norm, 
        levels=abs_diff_levels, cmap=cmap, extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{dist_data_label}-{ref_data_label}\nar_pr' + f' / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    cb1.ax.set_xticks(
        np.arange(abs_diff_levels[0], abs_diff_levels[-1]+np.diff(abs_diff_levels)[-1], 
                  (abs_diff_levels[-1]-abs_diff_levels[0])/5))
    ax2 = fig.add_subplot(132, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.precip - ref_data.precip)/norm, 
        levels=abs_diff_levels, cmap=cmap, extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label}-{ref_data_label}\nprecip' + f' / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    cb2.ax.set_xticks(
        np.arange(abs_diff_levels[0], abs_diff_levels[-1]+np.diff(abs_diff_levels)[-1], 
                  (abs_diff_levels[-1]-abs_diff_levels[0])/5))
    # Abs. diff
    norm = dT / 100
    unit = '%'
    ax3 = fig.add_subplot(133, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, dist_data.lat, 
        (dist_data.ar_pr - ref_data.ar_pr)/(dist_data.precip - ref_data.precip)/norm, 
        levels=rel_diff_levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label}\n' + '$\Delta$ar_pr / $\Delta$precip' + ' / % K$^{-1}$', 
        orientation='horizontal', ax=ax3, pad=0.05, shrink=0.6)
    cb3.ax.set_xticks(np.arange(rel_diff_levels[0], rel_diff_levels[-1]+np.diff(rel_diff_levels)[-1], 
                                (rel_diff_levels[-1]-rel_diff_levels[0])/5))
    axs = [ax1, ax2, ax3]
    return fig, axs

def plot_clim_mean_evap_diff_maps(ref_data, dist_data, ref_data_label, dist_data_label, dT):
    levels = np.arange(-2, 9, 1)
    cmap = 'viridis'
    norm = (86400)**-1
    unit = 'kg m$^{-2}$ day$^{-1}$'
    if dist_data_label == 'delta_HX':
        levels = np.arange(-0.3, 0.32, 0.02)
        cmap = 'coolwarm' 
        norm = dT/86400
        unit = 'kg m$^{-2}$ day$^{-1}$ K$^{-1}$'
    
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.evap/norm, 
        levels=levels, cmap=cmap, extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} evap / {unit}', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.evap/norm, 
        levels=levels, cmap=cmap, extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} evap / {unit}', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = dT/86400
    levels = np.arange(-0.3, 0.32, 0.02)
    unit = 'kg m$^{-2}$ day$^{-1}$ K$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.evap - ref_data.evap)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} evap / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6, format=tkr.FormatStrFormatter('%.1f'))
    # cb3.ax.set_xticklabels([round(l, 1) for l in levels])
    norm = dT*ref_data.evap / 100
    levels = np.arange(-15, 16, 1)
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.evap - ref_data.evap)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} evap '+'/ % K$^{-1}$', 
        orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    unit = 'kg m$^{-2}$ day$^{-1}$'
    ax1.set_title(f'{str(np.round(ref_data.evap.mean().values*86400, 2))} {unit}')
    ax2.set_title(f'{str(np.round(dist_data.evap.mean().values*86400, 2))} {unit}')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_rv_d_h2o_diff_maps(
        ref_data, dist_data, ref_data_label, dist_data_label, dT):
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
    norm = dT
    levels = np.arange(-20, 22, 2)
    unit = 'mm day$^{-1}$ K$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.rv_d_h2o - ref_data.rv_d_h2o)*86400/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} rv_d_h2o / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = dT*ref_data.rv_d_h2o / 100
    levels = np.arange(-10, 11, 1)
    unit = '% K$^{-1}$'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.rv_d_h2o - ref_data.rv_d_h2o)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} rv_d_h2o / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{str(np.round(ref_data.rv_d_h2o.sum().values*86400, 2))} mm day$^{{{-1}}}$')
    ax2.set_title(f'{str(np.round(dist_data.rv_d_h2o.sum().values*86400, 2))} mm day$^{{{-1}}}$')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs


def plot_clim_mean_diff_maps(
    ref_data, dist_data, ref_data_label, dist_data_label, variable, dT, 
    start_year, end_year, region='global'): 
    if region == 'NA':
        ref_data = du.sel_na(ref_data)
        dist_data = du.sel_na(dist_data)
    elif region == 'conus':
        ref_data = du.sel_conus_land(ref_data)
        dist_data = du.sel_conus_land(dist_data)

    if variable == 'ref_winds':
        ref_data['windspeed_ref'] = np.sqrt(ref_data.u_ref**2 + ref_data.v_ref**2)
        dist_data['windspeed_ref'] = np.sqrt(dist_data.u_ref**2 + dist_data.v_ref**2)
        fig, axs = plot_clim_mean_wind_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, dT, level='ref')
    if variable == '700_winds':
        ref_data['windspeed_700'] = np.sqrt(ref_data.u_700**2 + ref_data.v_700**2)
        dist_data['windspeed_700'] = np.sqrt(dist_data.u_700**2 + dist_data.v_700**2)
        fig, axs = plot_clim_mean_wind_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, dT, level='700')
    if variable == '250_winds':
        ref_data['windspeed_250'] = np.sqrt(ref_data.u_250**2 + ref_data.v_250**2)
        dist_data['windspeed_250'] = np.sqrt(dist_data.u_250**2 + dist_data.v_250**2)
        fig, axs = plot_clim_mean_wind_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, dT, level='250')
    if variable in ['precip', 'pr_intensity', 'prsn_intensity', 'ar_pr', 'ar_pr_intensity']:
        fig, axs = plot_clim_mean_precip_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, dT, variable)
    if variable in ['pr_frequency', 'prsn_frequency', 'ar_pr_frequency']:
        fig, axs = plot_clim_mean_pr_frequency_diff_map(
            ref_data, dist_data, ref_data_label, dist_data_label, dT, start_year, end_year, variable)
    if variable == 'olr':
        fig, axs = plot_clim_mean_olr_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, dT)
    if variable == 'evap':
        fig, axs = plot_clim_mean_evap_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, dT)
    if variable == 'rv_d_h2o':
        fig, axs = plot_clim_mean_rv_d_h2o_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, dT)
    if variable == 'ar_pr_to_pr_ratio':
        ref_data['ar_pr_to_pr_ratio'] = ref_data.ar_pr / ref_data.precip
        dist_data['ar_pr_to_pr_ratio'] = dist_data.ar_pr / dist_data.precip
        fig, axs = plot_clim_mean_ar_pr_to_pr_ratio_diff_map(
            ref_data, dist_data, ref_data_label, dist_data_label, dT, variable)
    if variable == 'ar_pr_diff_to_pr_diff_ratio':
        fig, axs = plot_clim_mean_ar_pr_diff_to_pr_diff_ratio_map(
            ref_data, dist_data, ref_data_label, dist_data_label, dT)
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

def plot_clim_mean_warming_trend_plots(
    exp_name_ctrl, exp_name_dist, 
    ctrl_label, dist_label, start_year, end_year,
    variables,
    regions=['conus', 'NA', 'global'],
    months=np.arange(1, 13), month_str='all_months',
    dd_exp_name_ctrl=None, dd_exp_name_dist=None,
    dd_ctrl_label=None, dd_dist_label=None,):
    missing_ts_all_vars = ['u_700', 'v_700', 'u_250', 'v_250', 
                                           'pr_intensity', 'prsn_intensity',
                                           'pr_frequency', 'prsn_frequency', 
                                           'ar_pr', 'ar_pr_intensity', 'ar_pr_frequency']
    plot_var_to_load_var_dict = {
        'ref_winds': {'atmos': ['u_ref', 'v_ref']},
        '700_winds': {'atmos': ['u_700', 'v_700']},
        '250_winds': {'atmos': ['u_250', 'v_250']},
        'precip': {'land': ['precip']},
        'pr_intensity': {'atmos_cmip': ['pr_intensity', 'prsn_intensity']},
        'pr_frequency': {'atmos_cmip': ['pr_frequency', 'prsn_frequency']},
        'ar_pr': {'atmos_cmip': ['ar_pr']},
        'ar_pr_intensity': {'atmos_cmip': ['ar_pr_intensity']},
        'ar_pr_frequency': {'atmos_cmip': ['ar_pr_frequency']},
        'ar_pr_to_pr_ratio': {'atmos_cmip': ['ar_pr'],
                              'land': ['precip']},
        'ar_pr_diff_to_pr_diff_ratio': {'atmos_cmip': ['ar_pr'],
                              'land': ['precip']},    
        'evap': {'land': ['evap_land']},
        'rv_d_h2o': {'river': ['rv_d_h2o']},
        'olr': {'atmos_cmip': ['olr']},
    }
    load_var_dict = {subset: [] for subset in ['atmos', 'atmos_cmip', 'land', 'river']}
    load_var_dict = {subset: [] for subset in ['atmos', 'atmos_cmip', 'land', 'river']}
    for var in variables:
        for subset, load_vars in plot_var_to_load_var_dict[var].items():
            for load_var in load_vars:
                load_var_dict[subset] += [load_var] if load_var not in load_var_dict[subset] else []
    load_var_dict = {subset: vars for subset, vars in load_var_dict.items() if vars != []}
    # Load clim. mean winds
    ts_all_path = {var: (exp_basepath_map[exp_name_ctrl]
                            if var not in missing_ts_all_vars
                            else '/archive/Marc.Prange/ts_all_missing_vars/')
                    for var in np.concatenate([var for var in load_var_dict.values()])           
                    }
    ref_paths = [glob(f'{ts_all_path[v]}{exp_name_ctrl}/ts_all/{sub}*.{v}.nc')[0] 
                    for sub in load_var_dict.keys()
                    for v in load_var_dict[sub]]
    ts_all_path = {var: (exp_basepath_map[exp_name_dist]
                            if var not in missing_ts_all_vars
                            else '/archive/Marc.Prange/ts_all_missing_vars/')
                    for var in np.concatenate([var for var in load_var_dict.values()])              
                    }
    dist_paths = [glob(f'{ts_all_path[v]}{exp_name_dist}/ts_all/{sub}*.{v}.nc')[0] 
                    for sub in load_var_dict.keys() 
                    for v in load_var_dict[sub]]
    print(f'Loading clim. mean data for simulation {ctrl_label}', flush=True)
    ref_data = du.lon_360_to_180(
            xr.open_mfdataset(ref_paths, compat='override', use_cftime=True).sel(
        time=slice(f'{start_year}', f'{end_year}')))
    ref_data_mean = ref_data.sel(time=ref_data.time.dt.month.isin(months)).mean('time')
    print(f'Loading clim. mean data for simulation {dist_label}', flush=True)
    dist_data = du.lon_360_to_180(
            xr.open_mfdataset(dist_paths, compat='override', use_cftime=True).sel(
        time=slice(f'{start_year}', f'{end_year}')))
    dist_data_mean = dist_data.sel(time=dist_data.time.dt.month.isin(months)).mean('time')
    for var in variables:
        if var in ['pr_frequency', 'prsn_frequency', 'ar_pr_frequency']:
            ref_data_mean[var] = ref_data[var].sum('time')
            dist_data_mean[var] = dist_data[var].sum('time')
    dT = du.get_global_mean_dT(exp_name_ctrl, exp_name_dist, start_year, end_year)
    if dd_exp_name_ctrl is not None:
        ts_all_path = {var: (exp_basepath_map[dd_exp_name_ctrl]
                            if var not in missing_ts_all_vars
                            else '/archive/Marc.Prange/ts_all_missing_vars/')
                    for var in np.concatenate([var for var in variables.values()])              
                    }
        dd_ref_paths = [glob(f'{ts_all_path[v]}{dd_exp_name_ctrl}/ts_all/{sub}*.{v}.nc')[0] 
                        for sub in variables.keys() 
                        for v in variables[sub]]
        ts_all_path = {var: (exp_basepath_map[dd_exp_name_dist]
                                if var not in missing_ts_all_vars
                                else '/archive/Marc.Prange/ts_all_missing_vars/')
                        for var in np.concatenate([var for var in variables.values()])              
                        }
        dd_dist_paths = [glob(f'{ts_all_path[v]}{dd_exp_name_dist}/ts_all/{sub}*.{v}.nc')[0] 
                        for sub in variables.keys() 
                        for v in variables[sub]]
        print(f'Loading clim. mean data for simulation {dd_ctrl_label}', flush=True)
        dd_ref_data = du.lon_360_to_180(xr.open_mfdataset(
            dd_ref_paths, compat='override').sel(
            time=slice(f'{start_year}', f'{end_year}')))
        dd_ref_data_mean = dd_ref_data.sel(time=dd_ref_data.time.dt.month.isin(months)).mean('time')
        print(f'Loading clim. mean data for simulation {dist_label}', flush=True)
        dd_dist_data = du.lon_360_to_180(xr.open_mfdataset(
            dd_dist_paths, compat='override').sel(
            time=slice(f'{start_year}', f'{end_year}')))
        dd_dist_data_mean = dd_dist_data.sel(time=dd_dist_data.time.dt.month.isin(months)).mean('time')
        for var in variables:
            if var in ['pr_frequency', 'prsn_frequency', 'ar_pr_frequency']:
                dd_ref_data_mean[var] = dd_ref_data[var].sum('time')
                dd_dist_data_mean[var] = dd_dist_data[var].sum('time')
        # Redefine dist and ref data means for double difference case
        # dist_data --> free-run difference
        # ref_data --> nudged-run difference
                        # HX_p2K            # HX_ctrl
        dist_data_mean = dist_data_mean - ref_data_mean.copy()
                        # nudge_p2K         # nudge_ctrl
        ref_data_mean = dd_dist_data_mean - dd_ref_data_mean
        # Change ctrl and dist labels accordingly
        ctrl_label = f'delta_{dd_ctrl_label[:-5]}'
        dist_label = 'delta_HX'

    plot_configs = {
    'region': regions,
    }
    for region in plot_configs['region']:
        for var in variables:
            print(
                f'Plotting {var} for simulations {dist_label}-{ctrl_label} over {region}.', 
                flush=True)
            fig, axs = plot_clim_mean_diff_maps(
                ref_data_mean, dist_data_mean, ctrl_label, dist_label, var, dT, start_year, end_year, region, )
            fig_dir = f'plots/clim_mean_warming_trend_plots/{dist_label}-{ctrl_label}/'
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                        fig_dir +
                        f'{ctrl_label}_{dist_label}_{start_year}-{end_year}_clim_mean_warming_trend_{var}_{region}_{month_str}.png',
                        dpi=300, bbox_inches='tight')


def _main():
    exp_name_ctrl = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min'
    exp_name_dist = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K'
    # dd_exp_name_ctrl = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day'
    # dd_exp_name_dist = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K'
    ctrl_label = 'nudge_30min_ctrl'
    dist_label = 'nudge_30min_p2K'
    # dd_ctrl_label = 'nudge_1day_ctrl'
    # dd_dist_label = 'nudge_1day_p2K'
    variables = ['ar_pr_frequency']
    start_year = 1951
    end_year = 2020

    plot_clim_mean_warming_trend_plots(
        exp_name_ctrl, exp_name_dist, 
        ctrl_label, dist_label, start_year, end_year,
        variables,
        regions=['conus'],
        months=np.arange(1, 13), 
        month_str='all_months'
        # dd_exp_name_ctrl, dd_exp_name_dist,
        # dd_ctrl_label, dd_dist_label
        )
        

if __name__ == '__main__':
    _main()
