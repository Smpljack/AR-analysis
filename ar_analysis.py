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
from skimage.measure import find_contours

import processing as arp

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


def _main():
    base_path = '/archive/Ming.Zhao/awg/2022.03/'
    year = 2016
    model_data = arp.load_model_data(
        base_path, year, variables=['prw', 'pr', 'prsn', 'ivtx', 'ivty'], 
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020', 
        ar_analysis=True)
    model_data_p2K = arp.load_model_data(
        base_path='/archive/Ming.Zhao/awg/2023.04/', year=year, 
        variables=['prw', 'pr', 'prsn', 'ivtx', 'ivty'], 
        exp_name='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K', 
        gfdl_processor='gfdl.ncrc5-intel23-classic-prod-openmp',
        ar_analysis=True)
    
    era5_data = arp.load_era5_data(
        base_path, year=2016, variables=['ivtx', 'ivty', 'prw'], 
        ar_analysis=True, mswep_precip=True, exp_name='c192_obs')
    model_data = arp.sel_na_pacific(model_data)
    model_data_p2K = arp.sel_na_pacific(model_data_p2K)
    era5_data = arp.sel_na_pacific(era5_data)
    # fig, ax = error_corr_map(model_data, era5_data)
    # fig.suptitle(f'{}', x=0.5, y=0.95)
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    # plt.savefig(
    #     f'/home/Marc.Prange/work/AR-analysis/plots/AR_ivt_prw_error_corr_map_01_2016.png', 
    #     dpi=300, bbox_inches='tight')

   
    for day in np.arange('2016-01-21', '2016-01-22', dtype='datetime64[D]'):
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131, projection=ccrs.PlateCarree())
        fig, ax1, c1 = plot_ivt_ar_shape(fig, ax1, model_data.sel(time=day, method='nearest').squeeze(), rain_contour=True)
        ax1.set(title='AM4')
        ax2 = fig.add_subplot(132, projection=ccrs.PlateCarree())
        fig, ax2, c2 = plot_ivt_ar_shape(fig, ax2, era5_data.sel(time=day, method='nearest').squeeze(), rain_contour=True)
        ax2.set(title='Observations (ERA5/MSWEP)')
        ax3 = fig.add_subplot(133, projection=ccrs.PlateCarree())
        fig, ax3, c3 = plot_ivt_ar_shape(fig, ax3, model_data_p2K.sel(time=day, method='nearest').squeeze(), rain_contour=True)
        ax3.set(title='AM4 with 2K warming')
        # fig, ax1, ax2, ax3, ax4 = plot_ivt_bias_rr(
        #     model_data.sel(time=day, method='nearest').squeeze(), 
        #     era5_data.sel(time=day, method='nearest').squeeze())
        fig.suptitle(f'{str(day)}', x=0.5, y=0.95)
        # plt.subplots_adjust(hspace=0.1, wspace=0.1)
        for axis in [ax1, ax2, ax3]:
            axis.set_extent([-170, -105, 20, 60], crs=ccrs.PlateCarree())
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
        norm= matplotlib.colors.Normalize(vmin=c3.cvalues.min(), vmax=c3.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=c3.cmap)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.1, 0.1, 0.7, 0.05])
        fig.colorbar(sm, cax=cbar_ax, label='Precipitation / mm day$^{-1}$', orientation='horizontal', ticks=c3.levels)
        plt.tight_layout()
        plt.savefig(
            f'/home/Marc.Prange/work/AR-analysis/plots/egu/AR_ivt_precip_map_ctrl_obs_warming_{str(day)}.pdf', 
            dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    _main()