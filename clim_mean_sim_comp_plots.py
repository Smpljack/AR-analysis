
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

def plot_clim_mean_precip_diff_maps(ref_data, dist_data, ref_data_label, dist_data_label, region='global'):
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.precip.values*86400,
        levels=np.arange(0, 10.5, 0.5), cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} pr '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.precip.values*86400, 
        levels=np.arange(0, 10.5, 0.5), cmap='viridis', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} pr '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    levels = np.arange(-1.5, 1.6, 0.1)
    unit = 'mm day$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.precip.values - ref_data.precip.values)*86400/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} pr / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = ref_data.precip.values / 100
    levels = np.arange(-50, 55, 5)
    unit = '%'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.precip.values - ref_data.precip.values)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} pr / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{np.round(ref_data.precip.mean().values*86400, 2)} mm day$^{{{-1}}}$')
    ax2.set_title(f'{np.round(dist_data.precip.mean().values*86400, 2)} mm day$^{{{-1}}}$')
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
    ref_data, dist_data, ref_data_label, dist_data_label, variable, region='global'): 
    if region == 'NA':
        ref_data = du.sel_na(ref_data)
        dist_data = du.sel_na(dist_data)

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
    if variable == 'precip':
        fig, axs = plot_clim_mean_precip_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region)
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
    ctrl_label, dist_label, start_year, end_year,
    variables, months=range(13), time_str='all_months'):
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
        ref_paths = [glob(f'{ts_all_path[v]}{exp_name_ctrl}/ts_all/mswep*.{v}.nc')[0]
                     for v in variables['mswep']]
    ts_all_path = {var: (exp_basepath_map[exp_name_dist]
                            if var not in ['u_700', 'v_700', 'u_250', 'v_250', 
                                           'pr_intensity', 'prsn_intensity', 'pr_frequency']
                            and (exp_name_dist != 'c192_obs')
                            else '/archive/Marc.Prange/ts_all_missing_vars/')
                    for var in np.concatenate([var for var in variables.values()])              
                    }
    dist_paths = [glob(f'{ts_all_path[v]}{exp_name_dist}/ts_all/{sub}*.{v}.nc')[0] 
                    for sub in variables.keys()
                    for v in variables[sub] if sub != 'mswep']
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
    
    plot_configs = {
    'region': ['global', 'NA'],
    }
    for region in plot_configs['region']:
        for var in np.concatenate([var for var in variables.values()]):
            print(
                f'Plotting {var} for simulations {dist_label}-{ctrl_label} over {region}.', 
                flush=True)
            fig, axs = plot_clim_mean_diff_maps(
                ref_data_mean, dist_data_mean, ctrl_label, dist_label, var, region, )
            fig_dir = f'plots/clim_mean_sim_comp/{dist_label}-{ctrl_label}/'
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                        fig_dir +
                        f'{ctrl_label}_{dist_label}_{start_year}-{end_year}_clim_mean_{var}_{region}_{time_str}.png',
                        dpi=300, bbox_inches='tight')


def _main():
    exp_name_ctrl = 'c192_obs'
    exp_name_dist = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min'
    ctrl_label = 'MSWEP'
    dist_label = 'nudge_30min_ctrl'
    start_year = 1979
    end_year = 2020
    variables = {
        'atmos_cmip': [], 
        'atmos': ['precip'], 
        'land': [], 
        'river': [], 
        'mswep': ['precip']}
    months = [11, 12, 1 , 2]
    plot_clim_mean_sim_comp(
        exp_name_ctrl, exp_name_dist, 
        ctrl_label, dist_label, start_year, end_year,
        variables, months, time_str='winter')
    
    

if __name__ == '__main__':
    _main()
