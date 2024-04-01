import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import ar_analysis as ara

def find_ar_days_at_coord(ar_data, lat, lon):
    ar_data_point = ar_data.sel({'lat': lat, 'lon': lon}, method='nearest')
    ar_times = ar_data_point.time.where(~np.isnan(ar_data_point.shape), drop=True)
    return ar_times

def _main():
    base_path = '/archive/Ming.Zhao/awg/2022.03/'
    exp_name_model = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020'
    exp_name_obs = 'c192_obs'
    years = np.arange(2010, 2021)
    loc_dict = {
        'lake_tahoe': 
        {
            'lat': 39.1,
            'lon': -120.0,
        },
        'sacramento':
        {
            'lat': 38.6,
            
            'lon': -121.5,
        }
    }
    model_data = xr.concat([ara.sel_california(ara.load_model_data(base_path, year, ['prw', 'pr', 'ivtx', 'ivty'], ar_analysis=True)) for year in years], dim='time')
    era5_data = xr.concat([ara.sel_california(ara.load_era5_data(base_path, year, ['ivtx', 'ivty', 'prw'], ar_analysis=True)) for year in years], dim='time')
    model_data = ara.sel_california(model_data)
    era5_data = ara.sel_california(era5_data)
    model_ar_data = xr.concat([ara.lon_360_to_180(ara.load_ar_data(base_path, exp_name_model, year)) for year in years], dim='time')
    era5_ar_data = xr.concat([ara.lon_360_to_180(ara.load_ar_data(base_path, exp_name_obs, year)) for year in years], dim='time')
    
   
    

    for loc, coords  in loc_dict.items():
        loc_ar_days_model = find_ar_days_at_coord(
            model_ar_data, coords['lat'], coords['lon'])
        loc_ar_days_model = np.unique(np.array(loc_ar_days_model, dtype='datetime64[1D]'))
        loc_ar_days_era5 = find_ar_days_at_coord(
            era5_ar_data, coords['lat'], coords['lon'])
        loc_ar_days_era5 = np.unique(np.array(loc_ar_days_era5, dtype='datetime64[1D]'))
        loc_dict[loc]['ar_days_era5'] = loc_ar_days_era5
        loc_dict[loc]['ar_days_model'] = loc_ar_days_model
    
    only_sacramento_bool = np.isin(loc_dict['sacramento']['ar_days_era5'], loc_dict['lake_tahoe']['ar_days_era5'], invert=True)
    tahoe_and_sacramento_bool = np.isin(loc_dict['lake_tahoe']['ar_days_era5'], loc_dict['sacramento']['ar_days_era5'])
    loc_dict['sacramento']['ar_days_era5'] = loc_dict['sacramento']['ar_days_era5'][only_sacramento_bool]
    loc_dict['lake_tahoe']['ar_days_era5'] = loc_dict['lake_tahoe']['ar_days_era5'][tahoe_and_sacramento_bool]

    only_sacramento_bool = np.isin(loc_dict['sacramento']['ar_days_model'], loc_dict['lake_tahoe']['ar_days_model'], invert=True)
    tahoe_and_sacramento_bool = np.isin(loc_dict['lake_tahoe']['ar_days_model'], loc_dict['sacramento']['ar_days_model'])
    loc_dict['sacramento']['ar_days_model'] = loc_dict['sacramento']['ar_days_model'][only_sacramento_bool]
    loc_dict['lake_tahoe']['ar_days_model'] = loc_dict['lake_tahoe']['ar_days_model'][tahoe_and_sacramento_bool]

    print('Plotting Histogram...')
    # Histogram
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121)
    months = np.arange('2016-01-01', '2017-02-01', dtype='datetime64[1M]')
    ax1.hist(loc_dict['lake_tahoe']['ar_days_model'], bins=months, histtype='bar', label='AM4', alpha=0.5)
    ax1.hist(loc_dict['lake_tahoe']['ar_days_era5'], bins=months, histtype='bar', label='ERA5', alpha=0.5)
    ax1.set_title('lake_tahoe')
    ax2 = fig.add_subplot(122)
    ax2.hist(loc_dict['sacramento']['ar_days_model'], bins=months, histtype='bar', label='AM4', alpha=0.5)
    ax2.hist(loc_dict['sacramento']['ar_days_era5'], bins=months, histtype='bar', label='ERA5', alpha=0.5)
    ax2.set_title('sacramento')
    ax2.legend()
    plt.savefig(
        f'/home/Marc.Prange/work/AR-analysis/plots/lake_tahoe_sacramento_comparison/AR_days_annual_cycle_hist_{years.min()}-{years.max()}.png', 
        bbox_inches='tight', dpi=300)
    print('Plotting maps...')
    # IVT/PRW bias maps
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)
    for (irow, (loc, coords))  in enumerate(loc_dict.items()):
        model_data_loc = model_data.sel(time=loc_dict[loc]['ar_days_model'], method='nearest').mean('time')
        era5_data_loc = era5_data.sel(time=loc_dict[loc]['ar_days_era5'], method='nearest').mean('time')

        ax1 = fig.add_subplot(gs[irow, 0], projection=ccrs.PlateCarree())
        cb_props = {
            'x': 0.15, 
            'y': 0.09,
            'w': 0.3,
            'h': 0.02,
            'label': r'$\frac{\Delta IVT}{IVT_{ERA5}} / -$',
            'vmin': -0.5,
            'vmax': 0.5,
            }
        fig, ax1 = ara.plot_bias_map(fig, ax1, model_data_loc, era5_data_loc, model_data_loc, 'ivt', rel_error=True, cb_props=cb_props)
        ax1.set_title(f'{loc} AR days model: {len(loc_dict[loc]["ar_days_model"])}', fontsize=8)
        ax2 = fig.add_subplot(gs[irow, 1], projection=ccrs.PlateCarree())
        cb_props = {
            'x': 0.57, 
            'y': 0.09,
            'w': 0.3,
            'h': 0.02,
            'label': r'$\frac{\Delta PRW}{PRW_{ERA5}} / -$',
            'vmin': -0.5,
            'vmax': 0.5,
            }
        fig, ax2 = ara.plot_bias_map(fig, ax2, model_data_loc, era5_data_loc, era5_data_loc, 'prw', rel_error=True, cb_props=cb_props)
        ax2.set_title(f'{loc} AR days ERA5: {len(loc_dict[loc]["ar_days_era5"])}', fontsize=8)
        for axis in [ax1, ax2]:
            axis.set_extent([era5_data.lon.min()+0.5, era5_data.lon.max()-0.5, era5_data.lat.min()+0.5, era5_data.lat.max()-0.5], crs=ccrs.PlateCarree())
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
            axis.scatter(loc_dict[loc]['lon'], loc_dict[loc]['lat'], marker='x', color='green')
            

    plt.savefig(f'/home/Marc.Prange/work/AR-analysis/plots/lake_tahoe_sacramento_comparison/AR_IVT_PRW_bias_maps_{years.min()}-{years.max()}.png', 
        dpi=300, bbox_inches='tight')
    

if __name__ == '__main__':
    _main()