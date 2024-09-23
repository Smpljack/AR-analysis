import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

import data_util as du


def plot_high_flow_stat_maps(
        HX_ctrl_count, HX_dist_count, nudged_ctrl_count, nudged_dist_count, 
        HX_ctrl_strength, HX_dist_strength, nudged_ctrl_strength, nudged_dist_strength, 
        threshold_kind='high', region='global', variable='flow'):
    if region == 'NA':
        HX_ctrl_count = du.sel_na(HX_ctrl_count)
        HX_dist_count = du.sel_na(HX_dist_count)
        nudged_ctrl_count = du.sel_na(nudged_ctrl_count)
        nudged_dist_count = du.sel_na(nudged_dist_count) 
        HX_ctrl_strength = du.sel_na(HX_ctrl_strength)
        HX_dist_strength = du.sel_na(HX_dist_strength)
        nudged_ctrl_strength = du.sel_na(nudged_ctrl_strength)
        nudged_dist_strength = du.sel_na(nudged_dist_strength) 
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(231, projection=ccrs.PlateCarree())
    levels = np.arange(-50, 55, 5)
    cf1 = ax1.contourf(
        HX_ctrl_count.lon, HX_ctrl_count.lat, 
        HX_dist_count - HX_ctrl_count, levels=levels, cmap='coolwarm', extend='both')
    cb1 = plt.colorbar(
        cf1, label=f'HX_p2K - HX_ctrl\n{threshold_kind} {variable} days '+ '/ -', orientation='horizontal', ax=ax1, 
        pad=0.05, shrink=0.6)
    ax2 = fig.add_subplot(232, projection=ccrs.PlateCarree())
    cf2 = ax2.contourf(
        nudged_ctrl_count.lon, nudged_ctrl_count.lat, 
        nudged_dist_count - nudged_ctrl_count, levels=levels, cmap='coolwarm', extend='both')
    cb2 = plt.colorbar(
        cf2, label=f'nudged_p2K - nudged_ctrl\n{threshold_kind} {variable} days '+ '/ -', orientation='horizontal', ax=ax2, 
        pad=0.05, shrink=0.6)
    ax3 = fig.add_subplot(233, projection=ccrs.PlateCarree())
    cf3 = ax3.contourf(
        nudged_ctrl_count.lon, nudged_ctrl_count.lat, 
        HX_dist_count - HX_ctrl_count - (nudged_dist_count - nudged_ctrl_count), levels=levels, cmap='coolwarm', extend='both')
    cb3 = plt.colorbar(
        cf3, label=f'$\Delta$HX-$\Delta$nudged\n{threshold_kind} {variable} days '+ '/ -', orientation='horizontal', ax=ax3, 
        pad=0.05, shrink=0.6)
    
    ax4 = fig.add_subplot(234, projection=ccrs.PlateCarree())
    levels = np.arange(-50, 55, 5)
    cf4 = ax4.contourf(
        HX_ctrl_strength.lon, HX_ctrl_strength.lat, 
        (HX_dist_strength - HX_ctrl_strength)/HX_ctrl_strength*100, 
        levels=levels, cmap='coolwarm', extend='both')
    cb4 = plt.colorbar(
        cf4, label=f'(HX_p2K - HX_ctrl)/HX_ctrl\n{threshold_kind} {variable} '+ '/ %', orientation='horizontal', ax=ax4, 
        pad=0.05, shrink=0.6)
    ax5 = fig.add_subplot(235, projection=ccrs.PlateCarree())
    cf5 = ax5.contourf(
        nudged_ctrl_strength.lon, nudged_ctrl_strength.lat, 
        (nudged_dist_strength - nudged_ctrl_strength)/nudged_ctrl_strength*100, 
        levels=levels, cmap='coolwarm', extend='both')
    cb5 = plt.colorbar(
        cf5, label=f'(nudged_p2K - nudged_ctrl)/nudged_ctrl\n{threshold_kind} {variable} '+ '/ %', orientation='horizontal', ax=ax5, 
        pad=0.05, shrink=0.6)
    ax6 = fig.add_subplot(236, projection=ccrs.PlateCarree())
    cf6 = ax6.contourf(
        nudged_ctrl_strength.lon, nudged_ctrl_strength.lat, 
        (HX_dist_strength - HX_ctrl_strength - (nudged_dist_strength - nudged_ctrl_strength))/HX_ctrl_strength*100, 
        levels=levels, cmap='coolwarm', extend='both')
    cb6 = plt.colorbar(
        cf6, label=f'($\Delta$HX-$\Delta$nudged)/HX_ctrl\n{threshold_kind} {variable} '+ '/ %', orientation='horizontal', ax=ax6, 
        pad=0.05, shrink=0.6)
    ax1.set_title(f'{np.round((HX_dist_count - HX_ctrl_count).mean().values, 2)}')
    ax2.set_title(f'{np.round((nudged_dist_count - nudged_ctrl_count).mean().values, 2)}')
    ax3.set_title(f'{np.round((HX_dist_count - HX_ctrl_count - (nudged_dist_count - nudged_ctrl_count)).mean().values, 2)}')
    ax4.set_title(f'{np.round(((HX_dist_strength - HX_ctrl_strength)/HX_ctrl_strength).mean().values*100, 2)} %')
    ax5.set_title(f'{np.round(((nudged_dist_strength - nudged_ctrl_strength)/HX_ctrl_strength).mean().values*100, 2)} %')
    ax6.set_title(f'{np.round(((HX_dist_strength - HX_ctrl_strength - (nudged_dist_strength - nudged_ctrl_strength))/HX_ctrl_strength).mean().values*100, 2)} %')
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    [ax.coastlines('50m') for ax in axs]
    return fig, axs


def _main():
    flow_stat_path = '/archive/Marc.Prange/discharge_statistics/'
    HX_ctrl_exp_name = 'c192L33_am4p0_amip_HIRESMIP_HX'
    HX_ctrl_label = 'HX_ctrl'
    HX_p2K_exp_name = 'c192L33_am4p0_amip_HIRESMIP_HX_p2K'
    HX_p2K_label = 'HX_p2K'
    nudged_ctrl_exp_name = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020'
    nudged_ctrl_label = 'nudge_6hr_ctrl'
    nudged_p2K_exp_name = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K'
    nudged_p2K_label = 'nudge_6hr_p2K'

    region = 'NA'
    threshold_kind = 'low'
    variable = 'flow'

    # Extreme flow event frequency
    HX_ctrl_event_count = xr.open_mfdataset(
        f'{flow_stat_path}/{HX_ctrl_exp_name}/{threshold_kind}_{variable}_count_monthly_{HX_ctrl_exp_name}_*.nc',
        concat_dim='year', combine='nested'
    ).to_array().sum('month').mean('year').squeeze()
    HX_p2K_event_count = xr.open_mfdataset(
        f'{flow_stat_path}/{HX_p2K_exp_name}/{threshold_kind}_{variable}_count_monthly_{HX_p2K_exp_name}_*.nc',
        concat_dim='year', combine='nested'
    ).to_array().sum('month').mean('year').squeeze()
    nudged_ctrl_event_count = xr.open_mfdataset(
        f'{flow_stat_path}/{nudged_ctrl_exp_name}/{threshold_kind}_{variable}_count_monthly_{nudged_ctrl_exp_name}_*.nc',
        concat_dim='year', combine='nested'
    ).to_array().sum('month').mean('year').squeeze()
    nudged_p2K_event_count = xr.open_mfdataset(
        f'{flow_stat_path}/{nudged_p2K_exp_name}/{threshold_kind}_{variable}_count_monthly_{nudged_p2K_exp_name}_*.nc',
        concat_dim='year', combine='nested'
    ).to_array().sum('month').mean('year').squeeze()
    # Extreme flow event strength
    HX_ctrl_event_strength = xr.open_dataarray(
        f'{flow_stat_path}/{HX_ctrl_exp_name}/{threshold_kind}_{variable}_threshold_{HX_ctrl_exp_name}_1980_2019.nc')
    HX_p2K_event_strength = xr.open_dataarray(
        f'{flow_stat_path}/{HX_p2K_exp_name}/{threshold_kind}_{variable}_threshold_{HX_p2K_exp_name}_1980_2019.nc')
    nudged_ctrl_event_strength = xr.open_dataarray(
        f'{flow_stat_path}/{nudged_ctrl_exp_name}/{threshold_kind}_{variable}_threshold_{nudged_ctrl_exp_name}_1980_2019.nc')
    nudged_p2K_event_strength = xr.open_dataarray(
        f'{flow_stat_path}/{nudged_p2K_exp_name}/{threshold_kind}_{variable}_threshold_{nudged_p2K_exp_name}_1980_2019.nc')

    fig, axs = plot_high_flow_stat_maps(
        HX_ctrl_event_count, HX_p2K_event_count, 
        nudged_ctrl_event_count, nudged_p2K_event_count, 
        HX_ctrl_event_strength, HX_p2K_event_strength, 
        nudged_ctrl_event_strength, nudged_p2K_event_strength,
        threshold_kind,
        region=region,
        variable=variable)
    fig.savefig(
        'plots/clim_mean_warming_trend_plots/high_low_flow_warming_trend_maps/'
        f'{threshold_kind}_{variable}_stat_map_warming_'
        f'{HX_ctrl_label}-{HX_p2K_label}-{nudged_ctrl_label}-{nudged_p2K_label}_{region}.png'
    )

if __name__ == '__main__':
    _main()
