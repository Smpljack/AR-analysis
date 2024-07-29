import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import os
from glob import glob
import pandas as pd

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
    ref_data, dist_data, ref_data_label, dist_data_label, 
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
        'prli_diff_rel': [-30, 30],
        # SNOW
        'prsn': [0, 15],
        'prsn_diff_abs': [-2, 2],
        'prsn_diff_rel': [-50, 50],
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
    dT = dist_data.ts - ref_data.ts
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
            z = (dist_data[var] - ref_data[var]) / dT / ref_std[var]
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
    start_year, end_year, abs_rel_diff, region='global', significance=True):
    if region == 'NA':
        ref_data = du.sel_na(ref_data)
        dist_data = du.sel_na(dist_data)
    n_days = len(np.arange(f"{start_year}-01-01", f"{end_year+1}-01-01", dtype='datetime64[D]'))
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    levels = np.linspace(0, 25, 20)
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        (ref_data.sum('time')/n_days)*100, 
        levels=levels, cmap='viridis', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} AR freq / %', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    cb1.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
    if abs_rel_diff == 'abs':
        norm = dT * n_days / 100
        levels = np.arange(-5, 5.5, 0.5)
    elif abs_rel_diff == 'rel':
        norm = dT * ref_data.sum('time') / 100
        levels = np.arange(-30, 33, 3)
    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        ref_data.lon, dist_data.lat, 
        (dist_data.sum('time') - ref_data.sum('time'))/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    c1 = ax2.contour(
        ref_data.lon, dist_data.lat, 
        (ref_data.sum('time')/n_days), 
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

def plot_clim_wind_diff_maps(
    ref_data, dist_data, ref_data_label, dist_data_label, region='global',): 
    if region == 'NA':
        ref_data = du.sel_na(ref_data)
        dist_data = du.sel_na(dist_data)
    ref_data['windspeed'] = np.sqrt(ref_data.u_ref**2 + ref_data.v_ref**2)
    dist_data['windspeed'] = np.sqrt(dist_data.u_ref**2 + dist_data.v_ref**2)

    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    cf1 = ax1.contourf(
        ref_data.lon, ref_data.lat, 
        ref_data.windspeed, 
        levels=np.arange(0, 7.65, 0.15), cmap='viridis', extend='both')
    if region == 'global':
        dq = 10
    elif region == 'NA':
        dq = 5
    qu1 = ax1.quiver(
        ref_data.lon[::dq], ref_data.lat[::dq], 
        ref_data.u_ref[::dq, ::dq], ref_data.v_ref[::dq, ::dq],
        )
    cb1 = plt.colorbar(
        cf1, label=f'{ref_data_label} 10 m winds '+ '/ m s$^{-1}$', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        dist_data.lon, dist_data.lat, 
        dist_data.windspeed, 
        levels=np.arange(0, 7.65, 0.15), cmap='viridis', extend='both')
    qu2 = ax2.quiver(
        dist_data.lon[::dq], dist_data.lat[::dq], 
        dist_data.u_ref[::dq, ::dq], dist_data.v_ref[::dq, ::dq],
        )
    cb2 = plt.colorbar(
        cf2, label=f'{dist_data_label} 10 m winds '+ '/ m s$^{-1}$', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    norm = 1
    levels = np.arange(-1, 1.1, 0.1)
    unit = 'm s$^{-1}$'
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.windspeed - ref_data.windspeed)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    qu3 = ax3.quiver(
        ref_data.lon[::5], ref_data.lat[::5], 
        dist_data.u_ref[::5, ::5] - ref_data.u_ref[::5, ::5], 
        dist_data.v_ref[::5, ::5] - ref_data.v_ref[::5, ::5],
        )
    cb3 = plt.colorbar(
        cf3, label=f'{dist_data_label}-{ref_data_label} 10 m winds / {unit}', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    norm = ref_data.windspeed / 100
    levels = np.arange(-50, 53, 5)
    unit = '%'
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    cf4 = ax4.contourf(
        ref_data.lon, ref_data.lat, 
        (dist_data.windspeed - ref_data.windspeed)/norm, 
        levels=levels, cmap='coolwarm', extend='both')
    qu4 = ax4.quiver(
        ref_data.lon[::5], ref_data.lat[::5], 
        dist_data.u_ref[::5, ::5] - ref_data.u_ref[::5, ::5], 
        dist_data.v_ref[::5, ::5] - ref_data.v_ref[::5, ::5],
        )
    cb4 = plt.colorbar(
        cf4, label=f'{dist_data_label}-{ref_data_label} 10 m winds / {unit}', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    # cb1.ax.set_xticks(np.arange(levels[0], levels[-1]+np.diff(levels)[-1], (levels[-1]-levels[0])/5))
    axs = [ax1, ax2, ax3, ax4]
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
    

def plot_basic_ar_stat_warming_maps(
    exp_name_ctrl, exp_name_warming, 
    ctrl_label, warming_label, start_year, end_year, base_path='/archive/Marc.Prange/ar_masked_monthly_data/', 
    min_precip=1, significance=1):
    data_ctrl_ar_masked_sum = du.load_ar_day_avg_stat(
        exp_name_ctrl, base_path, start_year, end_year, stat='sum', min_precip=min_precip)
    data_warming_ar_masked_sum = du.load_ar_day_avg_stat(
        exp_name_warming, base_path, start_year, end_year, stat='sum', min_precip=min_precip)
    # data_ctrl_ar_masked_sum['prli'] = data_ctrl_ar_masked_sum.pr - data_ctrl_ar_masked_sum.prsn
    # data_warming_ar_masked_sum['prli'] = data_warming_ar_masked_sum.pr - data_warming_ar_masked_sum.prsn
    data_ctrl_ar_count = du.load_ar_day_avg_stat(
        exp_name_ctrl, base_path, start_year, end_year, stat='count', min_precip=min_precip)
    data_warming_ar_count = du.load_ar_day_avg_stat(
        exp_name_warming, base_path, start_year, end_year, stat='count', min_precip=min_precip)
    data_ctrl_ar_day_mean = (data_ctrl_ar_masked_sum / data_ctrl_ar_count).mean('time')
    data_warming_ar_day_mean = (data_warming_ar_masked_sum / data_warming_ar_count).mean('time')
    if significance > 0:
        data_ctrl_ar_masked_std = du.load_ar_day_avg_stat(
            exp_name_ctrl, base_path, start_year, end_year, stat='std', min_precip=min_precip)
        # Get long-term standard deviation
        std_ctrl_long = std_long_from_std_monthly(
            data_ctrl_ar_masked_std, data_ctrl_ar_count, data_ctrl_ar_masked_sum / data_ctrl_ar_count)
        # Get long-term standard error used for z-test
        ste_ctrl_long= std_ctrl_long / np.sqrt(data_ctrl_ar_count.sum('time'))
    days_in_month = xr.DataArray(
        name='days_in_month',
        coords={'time': data_ctrl_ar_masked_sum.time.values}, 
        data=[pd.Period(str(date)).days_in_month for date in data_ctrl_ar_masked_sum.time.values])
    data_ctrl_ar_all_day_mean = (data_ctrl_ar_masked_sum / days_in_month).mean('time')
    data_warming_ar_all_day_mean = (data_warming_ar_masked_sum / days_in_month).mean('time')
    # Load clim. mean winds
    variables = ['u_ref', 'v_ref']
    ref_paths = [p for p in glob(f'/archive/Ming.Zhao/awg/2022.03/{exp_name_ctrl}/ts_all/*.nc')
                 if np.any([v in p for v in variables])]
    dist_paths = [p for p in glob(f'/archive/Ming.Zhao/awg/2022.03/{exp_name_warming}/ts_all/*.nc')
                 if np.any([v in p for v in variables])]
    ref_data_winds = du.lon_360_to_180(xr.open_mfdataset(ref_paths).sel(
        time=slice(f'{start_year}', f'{end_year}')).mean('time'))
    dist_data_winds = du.lon_360_to_180(xr.open_mfdataset(dist_paths).sel(
        time=slice(f'{start_year}', f'{end_year}')).mean('time'))

    # CREATE WINTER DATASETS? 
    plot_configs = {
        'variables': [['pr', 'prli', 'prsn'], ['mrro', 'rv_o_h2o', 'evap_land']],
        'weighting': ['ar_day', 'all_day'],
        'region': ['global', 'NA'],
        'abs_rel_diff': ['abs', 'rel'],
    }

    for region in plot_configs['region']:
        plot_clim_wind_diff_maps(
            ref_data_winds, dist_data_winds, ctrl_label, warming_label, region, )
        plt.savefig(
                    f'plots/ar_warming_stat_maps/{exp_name_warming}-{exp_name_ctrl}/'
                    f'{ctrl_label}_{warming_label}_clim_wind_ref_{region}.png',
                    dpi=300, bbox_inches='tight')
        for abs_rel_diff in plot_configs['abs_rel_diff']:
            dT = data_warming_ar_day_mean.ts - data_ctrl_ar_day_mean.ts
            fig, axs = plot_ar_frequency_diff_map(
                data_ctrl_ar_count, data_warming_ar_count, ctrl_label, warming_label, dT,
                start_year, end_year, abs_rel_diff, region=region, significance=True)
            plt.savefig(
                        f'plots/ar_warming_stat_maps/{exp_name_warming}-{exp_name_ctrl}/'
                        f'{ctrl_label}_{warming_label}_ar_freq_{abs_rel_diff}_{region}_{str(min_precip)}.png',
                        dpi=300, bbox_inches='tight')
            for variables in plot_configs['variables']:
                for weighting in plot_configs['weighting']:
                    if weighting == 'ar_day':
                        plot_ctrl_data = data_ctrl_ar_day_mean
                        plot_warming_data = data_warming_ar_day_mean
                    if weighting == 'all_day':
                        plot_ctrl_data = data_ctrl_ar_all_day_mean
                        plot_warming_data = data_warming_ar_all_day_mean
                    fig, axs = plot_AR_stat_diff_map(
                        ref_data=plot_ctrl_data, 
                        dist_data=plot_warming_data, 
                        ref_data_label=ctrl_label, 
                        dist_data_label=warming_label, 
                        plot_vars=variables, 
                        abs_rel_diff=abs_rel_diff, region=region, significance=significance, ref_std=ste_ctrl_long)
                    plt.savefig(
                        f'plots/ar_warming_stat_maps/{exp_name_warming}-{exp_name_ctrl}/'
                        f'{ctrl_label}_{warming_label}_{"_".join("%s" % "".join(x) for x in variables)}'
                        f'_{weighting}_{abs_rel_diff}_{region}_min_precip_{str(min_precip)}_significance_{significance}.png',
                        dpi=300, bbox_inches='tight')
    
def std_long_from_std_monthly(std_monthly, N_monthly, mean_monthly):
    """Calculate longterm 'mean' standard deviation from monthly stds.

    Args:
        std_monthly (xr.DataArray): Monthly std
        N_monthly (xr.DataArray): Monthly sample size
        mean_monthly (xr.DataArray): Monthly mean

    Returns:
        _type_: _description_
    """
    mean_long = (mean_monthly * N_monthly).sum('time') / N_monthly.sum('time')
    std_long = np.sqrt(
        ((std_monthly**2*(N_monthly-1)) + N_monthly*(mean_long - mean_monthly)**2).sum('time') / 
                (N_monthly.sum('time') - 1))
    return std_long


def _main():
    base_path = '/archive/Marc.Prange/ar_masked_monthly_data/'
    exp_name_ctrl = 'c192L33_am4p0_amip_HIRESMIP_HX'
    exp_name_dist = 'c192L33_am4p0_amip_HIRESMIP_HX_p2K'
    start_year = 1980
    end_year = 2019
    plot_basic_ar_stat_warming_maps(
        exp_name_ctrl, exp_name_dist, 'HX_ctrl', 'HX_p2K', 
        start_year, end_year, base_path, min_precip=1, significance=1.96)
    # variables = ['.u_ref.', '.v_ref.']
    # ref_paths = [p for p in glob(f'/archive/Marc.Prange/era5_monthly/*.nc')
    #             if np.any([v in p for v in variables])]
    # dist_paths = [p for p in glob(f'{base_path}{exp_name_dist}/ts_all/*.nc')
    #              if np.any([v in p for v in variables])]
    # ref_data_winds = du.lon_360_to_180(xr.open_mfdataset(ref_paths).sel(
    #     time=slice(f'{start_year}', f'{end_year}')).sel(
    #         time=slice(str(start_year), str(end_year))).mean('time'))
    # dist_data_winds = du.lon_360_to_180(xr.open_mfdataset(dist_paths).sel(
    #     time=slice(f'{start_year}', f'{end_year}')).sel(
    #         time=slice(str(start_year), str(end_year))).mean('time'))
    # dist_data_winds['lon'] = ref_data_winds['lon']
    # dist_data_winds['lat'] = ref_data_winds['lat']
    # plot_clim_wind_diff_maps(
    #         ref_data_winds, dist_data_winds, 'ERA5', 'nudged6hr_ctrl', 'global', )
    # plt.savefig(
    #             f'plots/ar_warming_stat_maps/{exp_name_dist}-{exp_name_ctrl}/'
    #             f'ERA5_HX_ctrl_clim_wind_ref_global.png',
    #             dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    _main()