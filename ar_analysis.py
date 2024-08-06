import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import os
from glob import glob
import pandas as pd
from pathlib import Path

import data_util as du

def error_corr_map(model_data, era5_data):
    ivt_abs_model = np.sqrt(model_data.ivtx**2 + model_data.ivty**2)
    ivt_abs_era5 = np.sqrt(era5_data.ivtx**2 + era5_data.ivty**2)
    ivt_error = (ivt_abs_model.where(np.logical_and(ivt_abs_model>100, model_data.ivtx>0))-
                 ivt_abs_era5.where(np.logical_and(ivt_abs_era5>100, era5_data.ivtx>0)))
    ivt_rel_error = ivt_error / ivt_abs_era5.where(np.logical_and(ivt_abs_era5>100, era5_data.ivtx>0))
    prw_error = (model_data.prw.where(np.logical_and(ivt_abs_model>100, model_data.ivtx>0))-
                 era5_data.prw.where(np.logical_and(ivt_abs_era5>100, era5_data.ivtx>0)))
    prw_rel_error = prw_error / era5_data.prw.where(np.logical_and(ivt_abs_era5>100, era5_data.ivtx>0))
    ivt_prw_rel_error_corr = xr.corr(ivt_rel_error, prw_rel_error, dim='time').values
    ivt_prw_error_corr = xr.corr(ivt_error, prw_error, dim='time').values

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    c1 = ax1.pcolormesh(
        ivt_error.lon, ivt_rel_error.lat, ivt_prw_error_corr, 
        vmin=-1, vmax=1, cmap='coolwarm')
    cax = fig.add_axes([0.15, 0.09, 0.3, 0.02])
    cb1 = fig.colorbar(
        c1, cax=cax, spacing='proportional', orientation='horizontal',
        extend='max', label='IVT-PRW abs. error correlation / -')
    
    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    c2 = ax2.pcolormesh(
        ivt_rel_error.lon, ivt_rel_error.lat, ivt_prw_rel_error_corr, 
        vmin=-1, vmax=1, cmap='coolwarm')
    cax = fig.add_axes([0.57, 0.09, 0.3, 0.02])
    cb1 = fig.colorbar(
        c2, cax=cax, spacing='proportional', orientation='horizontal',
        extend='max', label='IVT-PRW rel. error correlation / -')
    for axis in [ax1, ax2]:
        axis.set_extent([-170, -80, 10, 60], crs=ccrs.PlateCarree())
        axis.coastlines("10m", linewidth=0.5)
        # axis.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='tab:blue', edgecolor='tab:blue', linewidth=0.5)
        # axis.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='tab:blue', linewidth=0.5)
        axis.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5)
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        axis.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    return fig, ax1

def plot_ivt_ar_shape(fig, ax, data, rain_contour=False):
    ivt_abs = np.sqrt(data.ivtx**2 + data.ivty**2)
    c1 = ax.pcolormesh(
        data.lon, data.lat, data.ar_shape, cmap='Greys_r', alpha=0.2)
    if rain_contour:
        c2 = ax.contour(data.lon, data.lat, data.pr*86400, levels=np.arange(10, 80, 5), cmap='winter_r', linewidth=0.4)
    c3 = ax.quiver(
        data.lon[::4], data.lat[::4], 
        data.ivtx.where(np.logical_and(ivt_abs>100, data.ivtx>0))[::4, ::4], 
        data.ivty.where(np.logical_and(ivt_abs>100, data.ivtx>0))[::4, ::4], 
        animated=True, scale=1500, scale_units='inches', width=0.002, alpha=0.8)
    # qk = ax.quiverkey(c2, 0.7, 1.05, 250, r'250 $\mathrm{kg\,m\,s^{-1}}$', labelpos='E')
    return fig, ax, c2

def plot_bias_map(fig, ax, model_data, ref_data, quiver_data, var, cb_props, rel_error=False, mask_ar=False):
    if var == 'ivt':
        model_data['ivt'] = np.sqrt(model_data.ivtx**2 + model_data.ivty**2)
        ref_data['ivt'] = np.sqrt(ref_data.ivtx**2 + ref_data.ivty**2)
        quiver_data['ivt'] = np.sqrt(quiver_data.ivtx**2 + quiver_data.ivty**2)
    if mask_ar:
        model_data = model_data.where(np.logical_and(model_data.ivt>100, model_data.ivtx>0))
        ref_data = ref_data.where(np.logical_and(ref_data.ivt>100, ref_data.ivtx>0))
        quiver_data = quiver_data.where(np.logical_and(quiver_data.ivt>100, quiver_data.ivtx>0))
    error = model_data[var] - ref_data[var]
    if rel_error:
        error = error / ref_data[var]

    c = ax.pcolormesh(error.lon, error.lat, error, vmin=cb_props['vmin'], vmax=cb_props['vmax'], cmap='coolwarm')
    cax = fig.add_axes([cb_props['x'], cb_props['y'], cb_props['w'], cb_props['h']])
    cb1 = fig.colorbar(
        c, cax=cax, spacing='proportional', orientation='horizontal',
        extend='max', label=cb_props['label'])
    #$\Delta$ IVT / $\mathrm{kg\,m\,s^{-1}}$
    c = ax.quiver(
        quiver_data.lon[::4], quiver_data.lat[::4], 
        quiver_data.ivtx[::4, ::4], 
        quiver_data.ivty[::4, ::4], 
        animated=True, scale=1500, scale_units='inches', width=0.002, alpha=0.8)
    return fig, ax

def plot_ivt_bias_rr(model_data, era5_data):
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    fig, ax1 = plot_ivt_ar_shape(fig, ax1, model_data, rain_contour=True)

    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    fig, ax2 = plot_ivt_ar_shape(fig, ax2, era5_data, rain_contour=True)

    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cb_props = {
        'x': 0.15, 
        'y': 0.09,
        'w': 0.3,
        'h': 0.02,
        'label': r'$\frac{\Delta IVT}{IVT_{ERA5}} / -$',
        'vmin': -1,
        'vmax': 1,
    }
    fig, ax3 = plot_bias_map(
        fig, ax3, model_data, era5_data, model_data, 'ivt', rel_error=True, cb_props=cb_props, mask_ar=True)
    cb_props = {
        'x': 0.57, 
        'y': 0.09,
        'w': 0.3,
        'h': 0.02,
        'label': r'$\frac{\Delta PRW}{PRW_{ERA5}} / -$',
        'vmin': -1,
        'vmax': 1,
    }
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    fig, ax4 = plot_bias_map(
        fig, ax4, model_data, era5_data, era5_data, 'prw', rel_error=True, cb_props=cb_props, mask_ar=True)
        
    ivt_abs_model = np.sqrt(model_data.ivtx**2 + model_data.ivty**2)
    ivt_abs_era5 = np.sqrt(era5_data.ivtx**2 + era5_data.ivty**2)
    ivt_error = ivt_abs_model-ivt_abs_era5
    ivt_rel_error = ivt_error / ivt_abs_era5
    prw_error = model_data.prw-era5_data.prw
    prw_rel_error = prw_error / era5_data.prw
    ivt_prw_rel_error_corr = xr.corr(ivt_rel_error, prw_rel_error).values
    ivt_prw_error_corr = xr.corr(ivt_error, prw_error).values
    t2 = ax4.text(
        -150, 61, '(rel.) error corr(IVT, PRW): 'f'({np.round(ivt_prw_rel_error_corr, 2)}) {np.round(ivt_prw_error_corr, 2)}')
    for axis in [ax1, ax2, ax3, ax4]:
        axis.set_extent([-170, -80, 10, 60], crs=ccrs.PlateCarree())
        axis.coastlines("10m", linewidth=0.5)
        # axis.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='tab:blue', edgecolor='tab:blue', linewidth=0.5)
        # axis.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='tab:blue', linewidth=0.5)
        axis.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5)
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        axis.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax1.set(title='AM4')
    ax2.set(title='ERA5/MSWEP')

    return fig, ax1, ax2, ax3, ax4

def make_movie(image_folder, video_name, fps=2):
    images = np.sort(glob(image_folder))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

def plot_runoff_timeseries(model_data_loc, obs_data_loc):
    fig, axs = plt.subplots(nrows=5, sharex=True, figsize=(7, 9))
    axs[0].plot(model_data_loc.time, model_data_loc.pr, label='precip total')
    axs[0].plot(model_data_loc.time, model_data_loc.prsn, label='precip snow')
    axs[0].plot(obs_data_loc.time, obs_data_loc.pr, label='precip mswep', ls='-', lw=0.7)
    axs[0].set(ylabel='precip /\nmm day$^{-1}$')
    axs[0].legend()
    axs[1].plot(model_data_loc.time, model_data_loc.mrro, label='AM4/LM4.0')
    # axs[1].plot(minki_data.time, minki_data.runf, label='LM4.2-SHARC')
    axs[1].set(ylabel='total runoff /\nmm day$^{-1}$')
    axs[1].legend()
    h1, = axs[2].plot(model_data_loc.time, model_data_loc.snw)
    axs[2].set(ylabel='snow /\nkg m$^{-2}$')
    ax_twin = axs[2].twinx()
    h2, = ax_twin.plot(model_data_loc.time, model_data_loc.snw.differentiate('time', datetime_unit='D'), color='orange')
    ax_twin.set(ylabel=r'$\frac{d\mathrm{snow}}{dt}$ /'+'\nmm day$^{-1}$')
    axs[2].legend([h1, h2], ['snow', r'$\frac{d\mathrm{snow}}{dt}$'])
    axs[3].plot(model_data_loc.time, model_data_loc.evap_land)
    axs[3].set(ylabel=r'evap land /'+'\nmm day$^{-1}$')
    h1, = axs[4].plot(model_data_loc.time, model_data_loc.mrso - model_data_loc.mrso.mean('time'))
    axs[4].set(ylabel=r'$\mathrm{SM_{anom}}$ /'+'\nkg m$^{-2}$')
    ax_twin = axs[4].twinx()
    h2, = ax_twin.plot(model_data_loc.time, (model_data_loc.mrso - model_data_loc.mrso.mean('time')).differentiate('time', datetime_unit='D'), color='orange')
    ax_twin.set(ylabel=r'$\frac{d\mathrm{SM_{anom}}}{dt}$ /'+'\nmm day$^{-1}$')
    runoff = model_data_loc.mrro
    nf_precip = model_data_loc.pr - model_data_loc.prsn
    evap = model_data_loc.evap_land
    snow_melt = -(model_data_loc.snw.differentiate('time', datetime_unit='D') - model_data_loc.prsn)
    sm_budget = np.zeros(model_data_loc.mrso.shape)
    sm_budget[0] = model_data_loc.mrso[0]
    for i in range(1, len(sm_budget)):
        sm_budget[i] = (nf_precip - evap - runoff + snow_melt)[i-1] + sm_budget[i-1]
    h3, = axs[4].plot(model_data_loc.time, sm_budget - model_data_loc.mrso.mean('time').values, ls='--')
    axs[4].legend([h1, h2, h3], [r'$\mathrm{SM_{anom}}$', r'$\frac{d\mathrm{SM_{anom}}}{dt}$', '(P+SM-E-R)$\Delta$t'])
    for ax in axs:
        ylims = ax.get_ylim()
        ax.fill_between(model_data_loc.time, ylims[0], ylims[1]/2, where=model_data_loc.ar_shape == 1, alpha=0.3)
        ax.fill_between(model_data_loc.time, ylims[1]/2, ylims[1], where=obs_data_loc.ar_shape == 1, alpha=0.3, color='C2')
    axs[4].tick_params(axis='x', labelrotation=45)
    return fig, axs

def plot_runoff_timeseries_sharc(sharc_data, obs_data_loc):
    fig, axs = plt.subplots(nrows=7, sharex=True, figsize=(7, 13))
    axs[0].plot(sharc_data.time, sharc_data.lprec, label='lprec')
    axs[0].plot(sharc_data.time, sharc_data.fprec, label='fprecip')
    axs[0].plot(obs_data_loc.time, obs_data_loc.pr, label='precip mswep', ls='-', lw=0.7)
    axs[0].set(ylabel='precip /\nmm day$^{-1}$')
    axs[0].legend()
    axs[1].plot(sharc_data.time, sharc_data.reach_discharge)
    axs[1].set(ylabel='reach discharge /\nmm day$^{-1}$')
    h1 = axs[2].plot(sharc_data.time, sharc_data.snow)
    axs[2].set(ylabel='snow /\nkg m$^{-2}$')
    ax_twin = axs[2].twinx()
    h2 = ax_twin.plot(sharc_data.time, sharc_data.snow.differentiate('time', datetime_unit='D'), ls='--')
    ax_twin.set(ylabel=r'$\frac{d\mathrm{snow}}{dt}$ /'+'\nmm day$^{-1}$')
    axs[2].legend([h1[0], h2[0]], ['snow', r'$\frac{d\mathrm{snow}}{dt}$'])
    h_tran = axs[3].plot(sharc_data.time, sharc_data.tran, label='transp')
    h_evap = axs[3].plot(sharc_data.time, sharc_data.evap, label='transp')

    axs[3].set(ylabel=r'evaptransp /'+'\nmm day$^{-1}$')
    axs[3].legend([h_tran[0], h_evap[0]], ['transp', 'evap'], fontsize=8)
    h1 = axs[4].contourf(
        sharc_data.time, sharc_data.zfull_soil, sharc_data.soil_liq.sel(ptid=1).T,
        cmap='Blues')
    axs[4].set_yscale('log')
    axs[4].invert_yaxis()
    axs[4].set(ylabel='soil depth / m')
    cax = fig.add_axes([0.92, 0.34, 0.01, 0.08])
    cb = plt.colorbar(h1, cax=cax, orientation='vertical', label='soil_liq /'+' kg m$^{-3}$')
    h1 = axs[5].contourf(
        sharc_data.time, sharc_data.zfull_soil, sharc_data.gtos.sel(ptid=1).T,
        cmap='Blues')
    axs[5].set_yscale('log')
    axs[5].invert_yaxis()
    axs[5].set(ylabel='soil depth / m')
    axs[5].tick_params(axis='x', labelrotation=45)
    cax = fig.add_axes([0.92, 0.23, 0.01, 0.08])
    cb = plt.colorbar(h1, cax=cax, orientation='vertical', label='gtos /'+' mm day$^{-1}$')
    h1 = axs[6].plot(
        sharc_data.time, sharc_data.gw_stream_flux)
    axs[6].set(ylabel='gw_stream_flux /\n' 'mm day$^{-1}$')
    axs[6].tick_params(axis='x', labelrotation=45)
    return fig, axs


def plot_AR_stat_diff_map(
    ref_data, dist_data, ref_data_label, dist_data_label, dT,
    plot_vars, abs_rel_diff, region='global', significance=1, ref_std=False):
    unit_str = {
        # PRECIP
        'pr': 'mm day$^{-1}$',
        'pr_diff_abs': 'mm day$^{-1}$ K$^{-1}$',
        'pr_diff_rel': '% K$^{-1}$',
        # RAIN
        'prli': 'mm day$^{-1}$',
        'prli_diff_abs': 'mm day$^{-1}$ K$^{-1}$',
        'prli_diff_rel': '% K$^{-1}$',
        # SNOW
        'prsn': 'mm day$^{-1}$',
        'prsn_diff_abs': 'mm day$^{-1}$ K$^{-1}$',
        'prsn_diff_rel': '% K$^{-1}$',
        # DISCHARGE
        'rv_o_h2o': 'mm day$^{-1}$',
        'rv_o_h2o_diff_abs': 'mm day$^{-1}$ K$^{-1}$',
        'rv_o_h2o_diff_rel': '% K$^{-1}$',
        # EVAPORATION
        'evap_land': 'mm day$^{-1}$',
        'evap_land_diff_abs': 'mm day$^{-1}$ K$^{-1}$',
        'evap_land_diff_rel': '% K$^{-1}$',
        # RUNOFF
        'mrro': 'mm day$^{-1}$',
        'mrro_diff_abs': 'mm day$^{-1}$ K$^{-1}$',
        'mrro_diff_rel': '% K$^{-1}$',
    }
    var_scaling = {
        # PRECIP
        'pr': 86400,
        'pr_diff_abs': 86400,
        'pr_diff_rel': 1,
        # RAIN
        'prli': 86400,
        'prli_diff_abs': 86400,
        'prli_diff_rel': 1,
        # SNOW
        'prsn': 86400,
        'prsn_diff_abs': 86400,
        'prsn_diff_rel': 1,
        # DISCHARGE
        'rv_o_h2o': 86400,
        'rv_o_h2o_diff_abs': 86400,
        'rv_o_h2o_diff_rel': 1, 
        # EVAPORATION
        'evap_land': 86400,
        'evap_land_diff_abs': 86400,
        'evap_land_diff_rel': 1,  
        # RUNOFF
        'mrro': 86400,
        'mrro_diff_abs': 86400,
        'mrro_diff_rel': 1,
    }
    var_ref_vmin_vmax = {
        # PRECIP
        'pr': [0, 15],
        'pr_diff_abs': [-2, 2],
        'pr_diff_rel': [-8, 8],
        # RAIN
        'prli': [0, 15],
        'prli_diff_abs': [-2, 2],
        'prli_diff_rel': [-10, 10],
        # SNOW
        'prsn': [0, 15],
        'prsn_diff_abs': [-2, 2],
        'prsn_diff_rel': [-10, 10],
        # DISCHARGE
        'rv_o_h2o': [0, 50],
        'rv_o_h2o_diff_abs': [-5, 5],
        'rv_o_h2o_diff_rel': [-50, 50],
        # EVAPORATION
        'evap_land': [0, 3],
        'evap_land_diff_abs': [-0.3, 0.3],
        'evap_land_diff_rel': [-20, 20],
        # RUNOFF
        'mrro': [0, 2.5],
        'mrro_diff_abs': [-0.3, 0.3],
        'mrro_diff_rel': [-30, 30],
    }
    var_cmap = {
        'pr': 'Blues',
        'prli': 'Blues',
        'prsn': 'Blues',
        'rv_o_h2o': 'Blues',
        'evap_land': 'Blues',
        'mrro': 'Blues'
    }
    if region == 'NA':
        ref_data = du.sel_na(ref_data)
        dist_data = du.sel_na(dist_data)
    if abs_rel_diff == 'abs':
        norm = dT
    nvars = len(plot_vars)
    fig = plt.figure(figsize=(10, 3*nvars))
    axs = []
    for ivar, var in enumerate(plot_vars):
        ivar += 1
        if abs_rel_diff == 'rel':
            norm = dT * ref_data[var] / 100
        ax1 = fig.add_subplot(int(f"{nvars}2{2*ivar-1}"), projection=ccrs.PlateCarree())
        levels = np.linspace(
                var_ref_vmin_vmax[var][0], var_ref_vmin_vmax[var][1], 20)
        cf1 = ax1.contourf(
            ref_data.lon, ref_data.lat, 
            ref_data[var]*var_scaling[var], 
            levels=levels, 
            cmap=var_cmap[var], extend='both')
        cb1 = plt.colorbar(
            cf1, label=f'{ref_data_label} {var} / {unit_str[var]}', shrink=0.6,
            orientation='horizontal', pad=0.05)
        cb1.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
        axs.append(ax1)
        
        if significance > 0:
            # Apply z-test
            z = (dist_data[var] - ref_data[var]) / ref_std[var]
            dist_data[var] = dist_data[var].where(np.abs(z) > significance)
            ref_data[var] = ref_data[var].where(np.abs(z) > significance)
        ax2 = fig.add_subplot(int(f"{nvars}2{2*ivar}"), projection=ccrs.PlateCarree())
        levels = np.linspace(
                var_ref_vmin_vmax[var+f'_diff_{abs_rel_diff}'][0],
                var_ref_vmin_vmax[var+f'_diff_{abs_rel_diff}'][1], 20)
        cf2 = ax2.contourf(
            ref_data.lon, ref_data.lat, 
            (dist_data[var] - ref_data[var])/norm*var_scaling[f'{var}_diff_{abs_rel_diff}'], 
            levels=levels, 
            cmap='coolwarm', extend='both')
        cb2 = plt.colorbar(
            cf2, 
            label=f'$\Delta_{{{abs_rel_diff}}}${dist_data_label}-{ref_data_label} {var} / ' 
                  f'{unit_str[var+f"_diff_{abs_rel_diff}"]}', shrink=0.6, pad=0.05,
            orientation='horizontal',)

        cb2.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
        axs.append(ax2)

    [ax.coastlines("50m", linewidth=0.5) for ax in axs]
    [ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5) for ax in axs]
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
    [ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5) for ax in axs]
    plt.tight_layout()
    return fig, axs

def plot_ar_frequency_diff_map(
    ref_data, dist_data, ref_data_label, dist_data_label, dT,
    start_year, end_year, abs_rel_diff, region='global', 
    significance=1, ref_std=False):
    if region == 'NA':
        ref_data = du.sel_na(ref_data)
        dist_data = du.sel_na(dist_data)
    if significance > 0:
        # Apply z-test
        z = (dist_data - ref_data) / ref_std
        dist_data = dist_data.where(np.abs(z) > significance)
        ref_data = ref_data.where(np.abs(z) > significance)
    n_days = len(np.arange(f"{start_year}-01-01", f"{end_year+1}-01-01", dtype='datetime64[D]'))
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    levels = np.linspace(0, 25, 20)
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        (ref_data/n_days)*100, 
        levels=levels, cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} AR freq / %', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    cb1.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
    if abs_rel_diff == 'abs':
        norm = dT * n_days / 100
        levels = np.arange(-1, 1.1, 0.1)
    elif abs_rel_diff == 'rel':
        norm = dT * ref_data / 100
        levels = np.arange(-30, 33, 3)
    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        ref_data.lon, dist_data.lat, 
        (dist_data - ref_data)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    c1 = ax2.contour(
        ref_data.lon, dist_data.lat, 
        (ref_data/n_days), 
        levels=5, colors='gray', linewidths=0.7)
    cb2 = plt.colorbar(
        cf2, label=f'$\Delta_{{{abs_rel_diff}}}${dist_data_label}-{ref_data_label}'
                     ' AR freq / % K$^{-1}$', 
        orientation='horizontal', ax=ax2, pad=0.05, shrink=0.6)
    cb2.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
    ax1.coastlines("50m", linewidth=0.5)
    ax2.coastlines("50m", linewidth=0.5)
    axs = [ax1, ax2]
    [ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5) for ax in axs]
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
    [ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5) for ax in axs]
    plt.tight_layout()
    return fig, axs

def plot_clim_mean_wind_diff_maps(
        ref_data, dist_data, ref_data_label, dist_data_label, region='global',
        level='ref'):
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
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
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
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
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
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
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
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
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_precip_diff_maps(ref_data, dist_data, ref_data_label, dist_data_label, region='global'):
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.precip*86400, 
        levels=np.arange(0, 10.5, 0.5), cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} pr '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.precip*86400, 
        levels=np.arange(0, 10.5, 0.5), cmap='viridis', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} pr '+ '/ mm day$^{-1}$', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    levels = np.arange(-1, 1.1, 0.1)
    unit = 'mm day$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.precip - ref_data.precip)*86400/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} pr / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = ref_data.precip / 100
    levels = np.arange(-30, 33, 3)
    unit = '%'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.precip - ref_data.precip)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} pr / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{np.round(ref_data.precip.mean().values*86400, 2)} mm day$^{{{-1}}}$')
    ax2.set_title(f'{np.round(dist_data.precip.mean().values*86400, 2)} mm day$^{{{-1}}}$')
    axs = [ax1, ax2, ax3, ax4]
    return fig, axs

def plot_clim_mean_olr_diff_maps(ref_data, dist_data, ref_data_label, dist_data_label, region='global'):
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.olr, 
        levels=np.arange(0, 310, 10), cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} olr '+ '/ W m$^{-2}$', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.olr, 
        levels=np.arange(0, 310, 10), cmap='viridis', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} olr '+ '/ W m$^{-2}$', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    levels = np.arange(-20, 22, 2)
    unit = 'W m$^{-2}$'
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
    unit = '%'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.olr - ref_data.olr)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} olr / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{str(np.round(ref_data.olr.mean().values, 2))} W m$^{{{-2}}}$')
    ax2.set_title(f'{str(np.round(dist_data.olr.mean().values, 2))} W m$^{{{-2}}}$')
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
    if variable == 'rv_d_h2o':
        fig, axs = plot_clim_mean_rv_d_h2o_diff_maps(
            ref_data, dist_data, ref_data_label, dist_data_label, region)
    [ax.coastlines("50m", linewidth=0.5) for ax in axs]
    [ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5) for ax in axs]
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
    [ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5) for ax in axs]
    plt.tight_layout()
    return fig, axs
    

def plot_basic_sim_comp_maps(
    exp_name_ctrl, exp_name_warming, 
    ctrl_label, warming_label, start_year, end_year, dT,
    base_path='/archive/Marc.Prange/ar_masked_monthly_data/', 
    min_precip=1, min_ARs=30, significance=1,
    exp_name_ctrl_dist=None, exp_name_warming_dist=None,
    plot_clim_mean_vars=[], 
    plot_ar_day_means=True,
    plot_ar_freq=True,
    ):
    if plot_ar_day_means or plot_ar_freq:
        data_ctrl_ar_count = xr.open_dataset(
                f'{base_path}{exp_name_ctrl}/ar_day_mean/'
                f'{exp_name_ctrl}_ar_count_min_precip_{min_precip}.{start_year}-{end_year}.nc'
            ).ar_count.sum('time')
        data_warming_ar_count = xr.open_dataset(
                f'{base_path}{exp_name_warming}/ar_day_mean/'
                f'{exp_name_warming}_ar_count_min_precip_{min_precip}.{start_year}-{end_year}.nc'
            ).ar_count.sum('time')
    if plot_ar_day_means:
        data_ctrl_ar_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_ctrl}/ar_day_mean/'
                f'{exp_name_ctrl}_ar_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
        data_warming_ar_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_warming}/ar_day_mean/'
                f'{exp_name_warming}_ar_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
        data_ctrl_ar_all_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_ctrl}/ar_all_day_mean/'
                f'{exp_name_ctrl}_ar_all_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
        data_warming_ar_all_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_warming}/ar_all_day_mean/'
                f'{exp_name_warming}_ar_all_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
        if exp_name_ctrl_dist is not None:
            data_ctrl_dist_ar_count = xr.open_dataset(
                f'{base_path}{exp_name_ctrl_dist}/ar_day_mean/'
                f'{exp_name_ctrl_dist}_ar_count_min_precip_{min_precip}.{start_year}-{end_year}.nc'
            ).ar_count.sum('time') 
            data_warming_dist_ar_count = xr.open_dataset(
                f'{base_path}{exp_name_warming_dist}/ar_day_mean/'
                f'{exp_name_warming_dist}_ar_count_min_precip_{min_precip}.{start_year}-{end_year}.nc'
            ).ar_count.sum('time')
            data_ctrl_dist_ar_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_ctrl_dist}/ar_day_mean/'
                f'{exp_name_ctrl_dist}_ar_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
            data_warming_dist_ar_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_warming_dist}/ar_day_mean/'
                f'{exp_name_warming_dist}_ar_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
            data_ctrl_dist_ar_all_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_ctrl_dist}/ar_all_day_mean/'
                f'{exp_name_ctrl_dist}_ar_all_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
            data_warming_dist_ar_all_day_mean = xr.open_mfdataset(
                f'{base_path}{exp_name_warming_dist}/ar_all_day_mean/'
                f'{exp_name_warming_dist}_ar_all_day_mean_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
            # Number of ARs in warming run without thermodyn. change
            data_warming_ar_count = \
                (data_warming_ar_count - 
                (data_warming_dist_ar_count - data_ctrl_dist_ar_count))
            data_warming_ar_day_mean = \
                (data_warming_ar_day_mean - # Free warming run
                (data_warming_dist_ar_day_mean - data_ctrl_dist_ar_day_mean)) # Thermodyn. response
            data_warming_ar_all_day_mean = \
                (data_warming_ar_all_day_mean - # Free warming run
                (data_warming_dist_ar_all_day_mean - data_ctrl_dist_ar_all_day_mean)) # Thermodyn. response
        if significance > 0:
            std_ctrl_long = xr.open_mfdataset(
                f'{base_path}{exp_name_ctrl}/ar_day_mean/'
                f'{exp_name_ctrl}_ar_day_std_min_precip_{min_precip}.{start_year}-{end_year}.*.nc'
            )
            # Get long-term standard error used for z-test
            ste_ctrl_long= std_ctrl_long / np.sqrt(data_ctrl_ar_count)
            ndays = (np.datetime64(f'{end_year+1}-01-01') - 
                    np.datetime64(f'{start_year}-01-01')).astype('timedelta64[D]')/np.timedelta64(1, 'D')
            ste_ctrl_long_all_day = std_ctrl_long / np.sqrt(ndays)

    if (min_ARs > 0) & (plot_ar_day_means or plot_ar_freq):
        AR_count_mask = data_ctrl_ar_count > min_ARs
        AR_count_mask &= data_warming_ar_count > min_ARs
        if exp_name_ctrl_dist is not None:
            AR_count_mask &= data_ctrl_dist_ar_count > min_ARs
            AR_count_mask &= data_warming_dist_ar_count > min_ARs
        data_ctrl_ar_count = data_ctrl_ar_count.where(AR_count_mask)
        data_warming_ar_count = data_warming_ar_count.where(AR_count_mask)
        data_ctrl_ar_day_mean = data_ctrl_ar_day_mean.where(AR_count_mask)
        data_warming_ar_day_mean = data_warming_ar_day_mean.where(AR_count_mask)
        if exp_name_ctrl_dist is not None:
            data_ctrl_dist_ar_count = data_ctrl_dist_ar_count.where(AR_count_mask)
            data_warming_dist_ar_count = data_warming_dist_ar_count.where(AR_count_mask)
            data_ctrl_dist_ar_day_mean = data_ctrl_dist_ar_day_mean.where(AR_count_mask)
            data_warming_dist_ar_day_mean = data_warming_dist_ar_day_mean.where(AR_count_mask)
    if plot_clim_mean_vars != []:
        # Load clim. mean winds
        variables = {'atmos': [], 'land': [], 'river': []}
        if 'ref_winds' in plot_clim_mean_vars:
            variables['atmos'] += ['u_ref', 'v_ref']
        if '700_winds' in plot_clim_mean_vars:
            variables['atmos'] += ['u_700', 'v_700']
        if '700_winds' in plot_clim_mean_vars:
            variables['atmos'] += ['u_250', 'v_250']
        if 'precip' in plot_clim_mean_vars:
            variables['atmos'] += ['precip']
        if 'olr' in plot_clim_mean_vars:
            variables['atmos'] += ['olr']
        if 'rv_d_h2o' in plot_clim_mean_vars:
            variables['river'] += ['rv_d_h2o']
        ts_all_path = {var: ('/archive/Ming.Zhao/awg/2022.03/' 
                             if var not in ['u_700', 'v_700', 'u_250', 'v_250']
                             else '/archive/Marc.Prange/ts_all_upper_winds/')
                       for var in np.concatenate([var for var in variables.values()])              
                       }
        ref_paths = [glob(f'{ts_all_path[v]}{exp_name_ctrl}/ts_all/{sub}*.{v}.nc')[0] 
                     for sub in variables.keys() 
                     for v in variables[sub]]
        if exp_name_warming == 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K':
            ts_all_path = {
                var: (value if value != '/archive/Ming.Zhao/awg/2022.03/' else '/archive/Ming.Zhao/awg/2023.04/')
                for var, value in ts_all_path.items()}
        dist_paths = [glob(f'{ts_all_path[v]}{exp_name_warming}/ts_all/{sub}*.{v}.nc')[0] 
                      for sub in variables.keys() 
                      for v in variables[sub]]
        ref_data_mean = du.lon_360_to_180(xr.open_mfdataset(ref_paths, compat='override').sel(
            time=slice(f'{start_year}', f'{end_year}')).mean('time'))
        dist_data_mean = du.lon_360_to_180(xr.open_mfdataset(dist_paths, compat='override').sel(
            time=slice(f'{start_year}', f'{end_year}')).mean('time'))

    plot_configs = {
        'variables': [['pr', 'prli', 'prsn'], ],#['mrro', 'rv_o_h2o', 'evap_land']],
        'weighting': ['ar_day', 'all_day'],
        'region': ['global', 'NA'],
        'abs_rel_diff': ['abs', 'rel'],
    }

    for region in plot_configs['region']:
        for var in plot_clim_mean_vars:
            fig, axs = plot_clim_mean_diff_maps(
                ref_data_mean, dist_data_mean, ctrl_label, warming_label, var, region, )
            fig_dir = f'plots/ar_warming_stat_maps/{warming_label}-{ctrl_label}/clim_mean_plots/'
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                        fig_dir +
                        f'{ctrl_label}_{warming_label}_{start_year}-{end_year}_clim_mean_{var}_{region}.png',
                        dpi=300, bbox_inches='tight')
        for abs_rel_diff in plot_configs['abs_rel_diff']:
            if plot_ar_freq:
                fig, axs = plot_ar_frequency_diff_map(
                    data_ctrl_ar_count, data_warming_ar_count, ctrl_label, warming_label, dT,
                    start_year, end_year, abs_rel_diff, region=region, significance=True)
                fig_dir = f'plots/ar_warming_stat_maps/{warming_label}-{ctrl_label}/'
                Path(fig_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(
                            fig_dir +
                            f'{ctrl_label}_{warming_label}_{start_year}-{end_year}_ar_freq_{abs_rel_diff}_{region}_min_precip_{str(min_precip)}.png',
                            dpi=300, bbox_inches='tight')
            for variables in plot_configs['variables']:
                for weighting in plot_configs['weighting']:
                    if plot_ar_day_means:
                        if weighting == 'ar_day':
                            plot_ctrl_data = data_ctrl_ar_day_mean
                            plot_warming_data = data_warming_ar_day_mean
                            ste = ste_ctrl_long
                        if weighting == 'all_day':
                            plot_ctrl_data = data_ctrl_ar_all_day_mean
                            plot_warming_data = data_warming_ar_all_day_mean
                            ste = ste_ctrl_long_all_day
                        fig, axs = plot_AR_stat_diff_map(
                            ref_data=plot_ctrl_data, 
                            dist_data=plot_warming_data, 
                            ref_data_label=ctrl_label, 
                            dist_data_label=warming_label, 
                            dT=dT,
                            plot_vars=variables, 
                            abs_rel_diff=abs_rel_diff, region=region, significance=significance, ref_std=ste)
                        fig_dir = f'plots/ar_warming_stat_maps/{warming_label}-{ctrl_label}/'
                        Path(fig_dir).mkdir(parents=True, exist_ok=True)
                        fig_filename = \
                                f'{ctrl_label}_{warming_label}_{start_year}-{end_year}_{"_".join("%s" % "".join(x) for x in variables)}' \
                                f'_{weighting}_{abs_rel_diff}_{region}_min_precip_{str(min_precip)}_significance_{significance}.png'
                        plt.savefig(fig_dir+fig_filename, dpi=300, bbox_inches='tight')
    
def is_winter(month):
    return (month >= 11) | (month <= 2)

def _main():
    base_path = '/archive/Marc.Prange/ar_masked_monthly_data/'
    exp_name_ctrl = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020'
    exp_name_dist = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K'
    start_year = 1980
    end_year = 2019

    dT = xr.open_dataset(
        f'/archive/Ming.Zhao/awg/2023.04/{exp_name_dist}/ts_all/'
        f'atmos.197901-202012.t_ref.nc').t_ref.mean() - \
        xr.open_dataset(
        f'/archive/Ming.Zhao/awg/2022.03/{exp_name_ctrl}/ts_all/'
        f'atmos.195101-202012.t_ref.nc').t_ref.mean()
    plot_basic_sim_comp_maps(
        exp_name_ctrl, exp_name_dist, 'nudged6hr_ctrl', 'nudged6hr_p2K', 
        start_year, end_year, dT, base_path, min_precip=1, min_ARs=30, significance=1, 
        # exp_name_ctrl_dist='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
        # exp_name_warming_dist='c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K',
        plot_clim_mean_vars=[],#'ref_winds', '700_winds', '250_winds'],
        plot_ar_day_means=True,
        plot_ar_freq=True,
        )
    # variables = ['.u_ref.', '.v_ref.']
    # ref_paths = [p for p in glob(f'{base_path}2022.03/{exp_name_ctrl}/ts_all/*.nc')
    #             if np.any([v in p for v in variables])]
    # dist_paths = [p for p in glob(f'{base_path}2022.03/{exp_name_dist}/ts_all/*.nc')
    #              if np.any([v in p for v in variables])]
    # ref_data_winds = du.lon_360_to_180(xr.open_mfdataset(ref_paths).sel(
    #     time=slice(f'{start_year}', f'{end_year}')))
    # ref_data_winds = ref_data_winds.sel(
    #         time=is_winter(ref_data_winds['time.month'])).mean('time')
    # dist_data_winds = du.lon_360_to_180(xr.open_mfdataset(dist_paths).sel(
    #     time=slice(f'{start_year}', f'{end_year}')))
    # dist_data_winds = dist_data_winds.sel(
    #         time=is_winter(dist_data_winds['time.month'])).mean('time')
    # dist_data_winds['lon'] = ref_data_winds['lon']
    # dist_data_winds['lat'] = ref_data_winds['lat']
    # plot_clim_wind_diff_maps(
    #         ref_data_winds, dist_data_winds, 'HX_ctrl', 'HX_p2K', 'global', )
    # plt.savefig(
    #             f'plots/ar_warming_stat_maps/{exp_name_dist}-{exp_name_ctrl}/'
    #             f'HX_p2K-HX_ctrl_clim_wind_ref_global_winter.png',
    #             dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    _main()
