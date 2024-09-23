import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from pathlib import Path
import cartopy.feature as cfeature

import data_util as du
import ar_analysis as ara

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
        # DISCHARGE TO OCEAN
        'rv_d_h2o': 86400,
        'rv_d_h2o_diff_abs': 86400,
        'rv_d_h2o_diff_rel': 1, 
        # HIGH/LOW FLOW 
        'high_flow_count': 1,
        'high_flow_count_diff_abs': 1,
        'high_flow_count_diff_rel': 1, 
        'low_flow_count': 1,
        'low_flow_count_diff_abs': 1,
        'low_flow_count_diff_rel': 1, 
        # EVAPORATION
        'evap_land': 86400,
        'evap_land_diff_abs': 86400,
        'evap_land_diff_rel': 1,  
        # TRANSPIRATION
        'transp': 86400,
        'transp_diff_abs': 86400,
        'transp_diff_rel': 1,  
        # RUNOFF
        'mrro': 86400,
        'mrro_diff_abs': 86400,
        'mrro_diff_rel': 1,
        # SOIL MOISTURE
        'mrso': 86400,
        'mrso_diff_abs': 86400,
        'mrso_diff_rel': 86400,
        # Ts
        'ts': 1,
        'ts_diff_abs': 1,
        'ts_diff_rel': 1,
        # SNOW COVER
        'snw': 1,
        'snw_diff_abs': 1,
        'snw_diff_rel': 1,
        # WATER STORAGE
        'LWS': 1,
        'LWS_diff_abs': 1,
        'LWS_diff_rel': 1,
        'LWSv': 1,
        'LWSv_diff_abs': 1,
        'LWSv_diff_rel': 1,
        'FWS': 1,
        'FWS_diff_abs': 1,
        'FWS_diff_rel': 1,
        'FWSv': 1,
        'FWSv_diff_abs': 1,
        'FWSv_diff_rel': 1,
        # MELT
        'melt': 86400,
        'melt_diff_abs': 86400,
        'melt_diff_rel': 86400,
        'melts': 86400,
        'melts_diff_abs': 86400,
        'melts_diff_rel': 86400,
    }

unit_str = {
        # PRECIP
        'pr': 'kg day$^{-1}$',
        'pr_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'pr_diff_rel': '% K$^{-1}$',
        # RAIN
        'prli': 'kg day$^{-1}$',
        'prli_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'prli_diff_rel': '% K$^{-1}$',
        # SNOW
        'prsn': 'kg day$^{-1}$',
        'prsn_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'prsn_diff_rel': '% K$^{-1}$',
        # DISCHARGE
        'rv_o_h2o': 'kg day$^{-1}$',
        'rv_o_h2o_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'rv_o_h2o_diff_rel': '% K$^{-1}$',
        # DISCHARGE TO OCEAN
        'rv_d_h2o': 'kg day$^{-1}$',
        'rv_d_h2o_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'rv_d_h2o_diff_rel': '% K$^{-1}$',
        # HIGH/LOW FLOW
        'high_flow_count': 'days month$^{-1}$',
        'high_flow_count_diff_abs': 'days month$^{-1}$',
        'high_flow_count_diff_rel': '% month$^{-1}$',
        'low_flow_count': 'days month$^{-1}$',
        'low_flow_count_diff_abs': 'days month$^{-1}$',
        'low_flow_count_diff_rel': '% month$^{-1}$',
        # EVAPORATION
        'evap_land': 'kg day$^{-1}$',
        'evap_land_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'evap_land_diff_rel': '% K$^{-1}$',
        # TRANSPIRATION
        'transp': 'kg day$^{-1}$',
        'transp_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'transp_diff_rel': '% K$^{-1}$',
        # RUNOFF
        'mrro': 'kg day$^{-1}$',
        'mrro_diff_abs': 'kg day$^{-1}$ K$^{-1}$',
        'mrro_diff_rel': '% K$^{-1}$',
        # SOIL MOISTURE
        'mrso': 'kg day$^{-1}$',
        'mrso_diff_abs': 'kg day$^{-1}$',
        'mrso_diff_rel': 'kg day$^{-1}$',
        # Ts
        'ts': '° C',
        'ts_diff_abs': '° C',
        'ts_diff_rel': '° C',
        # SNOW COVER
        'snw': 'kg m$^{-2}$',
        'snw_diff_abs': 'kg m$^{-2}$',
        'snw_diff_rel': 'kg m$^{-2}$',
        # WATER STORAGE
        'LWS': 'kg m$^{-2}$',
        'LWS_diff_abs': 'kg m$^{-2}$',
        'LWS_diff_rel': 'kg m$^{-2}$',
        'LWSv': 'kg m$^{-2}$',
        'LWSv_diff_abs': 'kg m$^{-2}$',
        'LWSv_diff_rel': 'kg m$^{-2}$',
        'FWS': 'kg m$^{-2}$',
        'FWS_diff_abs': 'kg m$^{-2}$',
        'FWS_diff_rel': 'kg m$^{-2}$',
        'FWSv': 'kg m$^{-2}$',
        'FWSv_diff_abs': 'kg m$^{-2}$',
        'FWSv_diff_rel': 'kg m$^{-2}$',
        # MELT
        'melt': 'kg day$^{-1}$',
        'melt_diff_abs': 'kg day$^{-1}$',
        'melt_diff_rel': 'kg day$^{-1}$',
        'melts': 'kg day$^{-1}$',
        'melts_diff_abs': 'kg day$^{-1}$',
        'melts_diff_rel': 'kg day$^{-1}$',
    }

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def plot_labelled_basin_map(
        rv_basin_cubic, rv_basin_area_cubic, lon, lat, 
        min_basin_area=50000*1e6, outlet_markers=True):
    basin_area_dict = {
        int(basin): float(rv_basin_area_cubic.where(rv_basin_cubic == basin).sum().values) 
        for basin in list(np.unique(rv_basin_cubic)[:-1])}
    major_basins = [basin for basin in basin_area_dict.keys() if basin_area_dict[basin] > min_basin_area]
    rv_basin_cubic = rv_basin_cubic.where(np.isin(rv_basin_cubic, major_basins))
    cmap = rand_cmap(
        int(rv_basin_cubic.max().values), type='soft', first_color_black=True, last_color_black=False, verbose=True)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.pcolor(
        lon, lat, rv_basin_cubic, cmap=cmap,
    )
    if outlet_markers:
        rv_d_h2o = xr.open_mfdataset(
            '/archive/Marc.Prange/all_day_means/c192L33_am4p0_amip_HIRESMIP_HX/'
            'c192L33_am4p0_amip_HIRESMIP_HX_all_day_monthly_mean.1980-2019.rv_d_h2o.tile*.nc',
            concat_dim='grid_yt', combine='nested'
        ).rv_d_h2o.isel(month=0)
        rv_d_h2o['grid_yt'] = np.arange(1, len(rv_d_h2o.grid_yt)+1)
        for basin in major_basins:
            outlet_lons, outlet_lats = (
                lon.where(
                    (rv_basin_cubic == basin) & (rv_d_h2o > 0)
                    ).stack({'flat': ['grid_yt', 'grid_xt']}).dropna('flat', how='all').values,
                lat.where(
                    (rv_basin_cubic == basin) & (rv_d_h2o > 0)
                    ).stack({'flat': ['grid_yt', 'grid_xt']}).dropna('flat', how='all').values,)
            ax.scatter(outlet_lons, outlet_lats, marker='x', color='red')
    basin_mean_lon_lat = {
        basin: (lon.where(rv_basin_cubic == basin).mean(), lat.where(rv_basin_cubic == basin).mean()) 
        for basin in major_basins}
    [ax.annotate(f'{basin}', xy=(lon, lat), weight='bold')
     for basin, (lon, lat) in basin_mean_lon_lat.items()
     if (lat < 70) & (lat > 20) * (lon > -140) & (lon < -60)]
    ax.set_extent(
        [-140, -60, 20, 70], 
        crs=ccrs.PlateCarree())
    ax.coastlines()
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='tab:gray', edgecolor='tab:gray', linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='tab:blue', linewidth=1)
    fig.savefig(
        'plots/seasonal_cycle_for_regions/'
        'labelled_basin_map.png',
        dpi=300, bbox_inches='tight')

def plot_ar_count_seasonal_cycle_for_region(
        exp_names, exp_name_labels, base_path, start_year, end_year,
        region_mask, region_label, min_precip=1):
    fig = plt.figure(figsize=(4, 2))
    
    ax = fig.add_subplot(111)
    for exp_name, label in zip(exp_names, exp_name_labels):
        monthly_mean = xr.open_dataset(
            f'{base_path}ar_masked_monthly_data/{exp_name}/ar_day_mean/'
            f'{exp_name}_ar_count_min_precip_{min_precip}.{start_year}-{end_year}.nc'
        ).where(region_mask).groupby('time.month').mean()
        monthly_mean_region = monthly_mean.mean(['lat', 'lon'])
        ax.plot(
            monthly_mean_region.month, 
            monthly_mean_region.ar_count, 
            label=label)
        ax.set(xlabel='month', ylabel=f'AR count / -')
    ax.legend()
    fig.suptitle(f'{region_label}, {start_year}-{end_year}')
    var_str = 'ar_count'
    exp_str = "_".join("%s" % "".join(exp_label) for exp_label in exp_name_labels)
    fig.savefig(
        f'plots/seasonal_cycle_for_regions/{region_label.replace(" ", "_")}/'
        f'seasonal_cycle_{region_label.replace(" ", "_")}_{exp_str}_'
        f'{start_year}_{end_year}_{var_str}.png',
        dpi=300, bbox_inches='tight')
    
def plot_seasonal_cycle_for_region(
        exp_names, exp_name_labels, base_path, variables, start_year, end_year,
        region_mask, region_mask_cubic, region_label, 
        mean_types=['ar_day', 'ar_all_day'], min_precip=1, 
        land_area=None, land_area_cubic=None, cell_area=None, mrso_sat=None):
    """_summary_

    Args:
        exp_name (_type_): _description_
        base_path (_type_): _description_
        variable (_type_): _description_
    """

    fig = plt.figure(figsize=(4*len(mean_types), 2*len(variables)))
    axs = []
    nsubplot = 0
    for var in variables:
        for mean_type in mean_types:
            nsubplot += 1
            ax = fig.add_subplot(len(variables), len(mean_types), nsubplot)
            for exp_name, label in zip(exp_names, exp_name_labels):
                if 'ar' in mean_type:
                    mean_path = f'ar_masked_monthly_data/{exp_name}/{mean_type}_mean/'
                    min_precip_str = f'_min_precip_{min_precip}'
                else:
                    mean_path = f'all_day_means/{exp_name}/'
                    min_precip_str = ''
                print(
                    f"Plotting {mean_type} seasonal cycle of "
                    f"{var} for {region_label} \n for sim {label}")
                if var == 'rv_d_h2o' and mean_type == 'all_day':
                    monthly_mean = xr.open_mfdataset(
                    f'{base_path}{mean_path}'
                    f'{exp_name}_{mean_type}_monthly_mean'
                    f'{min_precip_str}.{start_year}-{end_year}.{var}.tile*.nc',
                    concat_dim='grid_yt', combine='nested')
                    monthly_mean['grid_yt'] = np.arange(1, len(monthly_mean.grid_yt)+1)
                else:
                    monthly_mean = xr.open_dataset(
                        f'{base_path}{mean_path}'
                        f'{exp_name}_{mean_type}_monthly_mean'
                        f'{min_precip_str}.{start_year}-{end_year}.{var}.nc'
                    )
                if 'ar' not in mean_type and var != 'rv_d_h2o':
                    monthly_mean = du.lon_360_to_180(monthly_mean)
                if var == 'rv_d_h2o' and mean_type == 'all_day':
                    land_area_cubic = land_area_cubic.where(region_mask_cubic)
                    monthly_mean_region = (monthly_mean.where(
                            region_mask_cubic)*land_area_cubic).sum(['grid_xt', 'grid_yt'])
                elif var in ['high_flow_count', 'low_flow_count'] and mean_type == 'all_day':
                    monthly_mean_region = monthly_mean.where(
                        region_mask).mean(['lat', 'lon'])
                else:
                    if var == 'rv_d_h2o':
                        land_area = land_area.where(region_mask)
                        monthly_mean_region = (monthly_mean.where(
                            region_mask)*land_area).sum(['lon', 'lat'])
                    elif var in ['evap_land', 'melt', 'melts', 'transp',
                                 'snw', 'LWS', 'FWS', 'LWSv', 'FWSv']:
                        land_area = land_area.where(region_mask)
                        monthly_mean_region = (monthly_mean * land_area).where(
                            region_mask).sum(['lon', 'lat']) 
                    elif var in ['pr', 'prli', 'prsn']:
                        cell_area = cell_area.where(region_mask)
                        monthly_mean_region = (monthly_mean * cell_area).where(
                            region_mask).sum(['lon', 'lat'])
                    elif var in ['mrso']:
                        sec_in_month = {month+1: days*86400
                             for month, days in 
                             enumerate([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])}
                        sec_in_month = xr.DataArray(
                            data=list(sec_in_month.values()), 
                            coords={'month': list(sec_in_month.keys())}, 
                            dims='month')
                        monthly_mean_region = monthly_mean_region.differentiate('month') / \
                            sec_in_month
                        monthly_mean_region = (monthly_mean.where(
                            region_mask)*land_area).sum(['lon', 'lat']) / \
                            (land_area).sum(['lon', 'lat'])
                    elif var in ['ts']:
                        monthly_mean_region = (monthly_mean.where(
                            region_mask)).mean(['lon', 'lat']) - 273.15
                ax.plot(
                    monthly_mean_region.month, 
                    monthly_mean_region[f'{var}']*var_scaling[var], 
                    label=label)

            ax.set(xlabel='month', ylabel=f'{var} /\n{unit_str[var]}')
            axs.append(ax)
    axs[0].legend()
    [axs[i].set_title(mean_type) 
     for i, mean_type in enumerate(mean_types)]
    fig.suptitle(f'{region_label}, {start_year}-{end_year}')
    var_str = "_".join("%s" % "".join(var) 
                       for var in variables)
    exp_str = "_".join("%s" % "".join(exp_label) 
                       for exp_label in exp_name_labels)
    plot_path = f'plots/seasonal_cycle_for_regions/{region_label.replace(" ", "_")}'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        f'{plot_path}/'
        f'seasonal_cycle_{region_label.replace(" ", "_")}_{exp_str}_'
        f'{start_year}_{end_year}_{var_str}.png',
        dpi=300, bbox_inches='tight')

def plot_therm_dyn_change_seasonal_cycle_for_region(
        exp_names, exp_name_labels, base_path, variables, start_year, end_year,
        region_mask, region_mask_cubic, region_label, 
        mean_types=['ar_day', 'ar_all_day'], min_precip=1, 
        land_area=None, land_area_cubic=None, cell_area=None, mrso_sat=None):
    """_summary_

    Args:
        exp_name (_type_): _description_
        base_path (_type_): _description_
        variable (_type_): _description_
    """

    fig = plt.figure(figsize=(4*len(mean_types), 2*len(variables)))
    axs = []
    nsubplot = 0
    for var in variables:
        for mean_type in mean_types:
            nsubplot += 1
            ax = fig.add_subplot(len(variables), len(mean_types), nsubplot)
            for exp_name, label in zip(exp_names, exp_name_labels):
                if 'ar' in mean_type:
                    mean_path = f'ar_masked_monthly_data/{exp_name}/{mean_type}_mean/'
                    min_precip_str = f'_min_precip_{min_precip}'
                else:
                    mean_path = f'all_day_means/{exp_name}/'
                    min_precip_str = ''
                print(
                    f"Plotting {mean_type} seasonal cycle of "
                    f"{var} for {region_label} \n for sim {label}")
                if var == 'rv_d_h2o' and mean_type == 'all_day':
                    monthly_mean = xr.open_mfdataset(
                    f'{base_path}{mean_path}'
                    f'{exp_name}_{mean_type}_monthly_mean'
                    f'{min_precip_str}.{start_year}-{end_year}.{var}.tile*.nc',
                    concat_dim='grid_yt', combine='nested')
                    monthly_mean['grid_yt'] = np.arange(1, len(monthly_mean.grid_yt)+1)
                else:
                    monthly_mean = xr.open_dataset(
                        f'{base_path}{mean_path}'
                        f'{exp_name}_{mean_type}_monthly_mean'
                        f'{min_precip_str}.{start_year}-{end_year}.{var}.nc'
                    )
                if 'ar' not in mean_type and var != 'rv_d_h2o':
                    monthly_mean = du.lon_360_to_180(monthly_mean)
                if var == 'rv_d_h2o' and mean_type == 'all_day':
                    land_area_cubic = land_area_cubic.where(region_mask_cubic)
                    monthly_mean_region = (monthly_mean.where(
                            region_mask_cubic)*land_area_cubic).sum(['grid_xt', 'grid_yt'])
                elif var in ['high_flow_count', 'low_flow_count'] and mean_type == 'all_day':
                    monthly_mean_region = monthly_mean.where(
                        region_mask).mean(['lat', 'lon'])
                else:
                    if var == 'rv_d_h2o':
                        land_area = land_area.where(region_mask)
                        monthly_mean_region = (monthly_mean.where(
                            region_mask)*land_area).sum(['lon', 'lat'])
                    elif var in ['evap_land', 'melt', 'melts', 'transp',
                                 'snw', 'LWS', 'FWS', 'LWSv', 'FWSv']:
                        land_area = land_area.where(region_mask)
                        monthly_mean_region = (monthly_mean * land_area).where(
                            region_mask).sum(['lon', 'lat']) 
                    elif var in ['pr', 'prli', 'prsn']:
                        cell_area = cell_area.where(region_mask)
                        monthly_mean_region = (monthly_mean * cell_area).where(
                            region_mask).sum(['lon', 'lat'])
                    elif var in ['mrso']:
                        sec_in_month = {month+1: days*86400
                             for month, days in 
                             enumerate([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])}
                        sec_in_month = xr.DataArray(
                            data=list(sec_in_month.values()), 
                            coords={'month': list(sec_in_month.keys())}, 
                            dims='month')
                        monthly_mean_region = monthly_mean_region.differentiate('month') / \
                            sec_in_month
                        monthly_mean_region = (monthly_mean.where(
                            region_mask)*land_area).sum(['lon', 'lat']) / \
                            (land_area).sum(['lon', 'lat'])
                    elif var in ['ts']:
                        monthly_mean_region = (monthly_mean.where(
                            region_mask)).mean(['lon', 'lat']) - 273.15
                ax.plot(
                    monthly_mean_region.month, 
                    monthly_mean_region[f'{var}']*var_scaling[var], 
                    label=label)

            ax.set(xlabel='month', ylabel=f'{var} /\n{unit_str[var]}')
            axs.append(ax)
    axs[0].legend()
    [axs[i].set_title(mean_type) 
     for i, mean_type in enumerate(mean_types)]
    fig.suptitle(f'{region_label}, {start_year}-{end_year}')
    var_str = "_".join("%s" % "".join(var) 
                       for var in variables)
    exp_str = "_".join("%s" % "".join(exp_label) 
                       for exp_label in exp_name_labels)
    plot_path = f'plots/seasonal_cycle_for_regions/{region_label.replace(" ", "_")}'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        f'{plot_path}/'
        f'seasonal_cycle_{region_label.replace(" ", "_")}_{exp_str}_'
        f'{start_year}_{end_year}_{var_str}.png',
        dpi=300, bbox_inches='tight')

    
def plot_water_balance_seasonal_cycle_for_region(
        exp_names, exp_name_labels, base_path, pos_variables, neg_variables, 
        start_year, end_year, region_mask, region_mask_cubic, region_label, 
        land_area=None, land_area_cubic=None, cell_area=None):
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111)
    sec_in_month = {month+1: days*86400
                             for month, days in 
                             enumerate([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])}
    sec_in_month = xr.DataArray(
        data=list(sec_in_month.values()), 
        coords={'month': list(sec_in_month.keys())}, 
        dims='month')
    for exp_name, label in zip(exp_names, exp_name_labels):
        monthly_means = {var: (du.lon_360_to_180(xr.open_dataset(
                                f'{base_path}all_day_means/{exp_name}/'
                                f'{exp_name}_all_day_monthly_mean'
                                f'.{start_year}-{end_year}.{var}.nc'))
                            if var != 'rv_d_h2o'
                            else xr.open_mfdataset(
                                f'{base_path}all_day_means/{exp_name}/'
                                f'{exp_name}_all_day_monthly_mean'
                                f'.{start_year}-{end_year}.{var}.tile*.nc',
                                concat_dim='grid_yt', combine='nested'))[var]
                        for var in pos_variables+neg_variables
                        }
        if 'rv_d_h2o' in monthly_means.keys():
            monthly_means['rv_d_h2o']['grid_yt'] = np.arange(1, len(monthly_means['rv_d_h2o'].grid_yt)+1)
        for var in monthly_means.keys():
            if var in ['mrso', 'snw', 'LWS', 'FWS', 'LWSv', 'FWSv']:
                monthly_means[var] = monthly_means[var].differentiate('month') / sec_in_month
        # Apply region mask
        monthly_means = {var: (monthly_mean.where(region_mask) 
                                   if var != 'rv_d_h2o'
                                   else monthly_mean.where(region_mask_cubic))
                             for var, monthly_mean in monthly_means.items()}
        land_area = land_area.where(region_mask)
        land_area_cubic = land_area_cubic.where(region_mask_cubic)
        cell_area = land_area.where(region_mask)
        # Calculate basin integral
        monthly_means = {var: (monthly_mean * land_area).sum(['lon', 'lat'])
                               for var, monthly_mean in monthly_means.items()
                        if var not in ['rv_d_h2o']}
        if 'rv_d_h2o' in monthly_means.keys():
            monthly_means['rv_d_h2o'] = (
                monthly_means['rv_d_h2o'] * land_area_cubic).sum(['grid_xt', 'grid_yt'])
        # Seperate positive and negative monthly means
        pos_monthly_means = [monthly_mean for var, monthly_mean in monthly_means.items() 
                             if var in pos_variables]
        neg_monthly_means = [monthly_mean for var, monthly_mean in monthly_means.items() 
                             if var in neg_variables]
        balance = (sum(pos_monthly_means) - sum(neg_monthly_means))*sec_in_month
        annual_balance = sum(balance).values
        ax.plot(np.arange(1, 13), balance, label=label)
    ax.set(xlabel='month', ylabel='water balance / kg month$^{-1}$')
    fig.legend(prop={'size': 4})
    pos_var_str = "+".join("%s" % "".join(var) 
                       for var in pos_variables)
    neg_var_str = "+".join("%s" % "".join(var) 
                       for var in neg_variables)
    ax.set_title(
        f"{region_label} water balance"
        f"\n{pos_var_str}-({neg_var_str})"
        f"\nAnnual balance: {annual_balance:.2e} "
         "kg year$^{-1}$", 
        fontdict={'fontsize': 8})
    plot_path = f'plots/seasonal_cycle_for_regions/{region_label.replace(" ", "_")}'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    exp_str = "_".join("%s" % "".join(exp_label)
                       for exp_label in exp_name_labels)
    fig.savefig(
        f'{plot_path}/'
        f'seasonal_cycle_balance_{region_label.replace(" ", "_")}_{exp_str}_'
        f'{start_year}_{end_year}_{pos_var_str}-{neg_var_str}.png',
        dpi=300, bbox_inches='tight')

def _main():
    exp_names = ['c192L33_am4p0_amip_HIRESMIP_HX', 
                #  'c192L33_am4p0_amip_HIRESMIP_HX_p2K',
                #  'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020',
                #  'c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K',
                 ]
    exp_name_labels = ['HX_ctrl', ]#'HX_p2K', 'nudged6hr_ctrl', 'nudged6hr_p2K']
    base_path = '/archive/Marc.Prange/'
    variables = ['rv_d_h2o', 'pr', 'evap_land', 'LWS', 'FWS', 'LWSv', 'FWSv']
    start_year = 1980
    end_year = 2019 
    region_label = 'sacramento river basin'
    basin_id = 104
    mean_types = ['all_day']
    min_precip = 1
    static_base_path = '/archive/Ming.Zhao/awg/2022.03/c192L33_am4p0_amip_HIRESMIP_HX/'\
                       'gfdl.ncrc4-intel-prod-openmp/pp'

    river_static = du.lon_360_to_180(
        xr.open_dataset(
            f'{static_base_path}/river/river.static.nc'))
    rv_static_cubic_t3 = xr.open_dataset(
        f'{static_base_path}/river_cubic/river_cubic.static.tile3.nc')
    rv_static_cubic_t5 = xr.open_dataset(
        f'{static_base_path}/river_cubic/river_cubic.static.tile5.nc')
    rv_static_cubic = xr.concat([rv_static_cubic_t3, rv_static_cubic_t5], dim='grid_yt')
    rv_static_cubic['grid_yt'] = np.arange(1, len(rv_static_cubic.grid_yt)+1)
    land_static = du.lon_360_to_180(xr.open_dataset(
        f'{static_base_path}/land/land.static.nc'))
    land_static_cubic_t3 = xr.open_dataset(
        f'{static_base_path}/land_cubic/land_cubic.static.tile3.nc')
    land_static_cubic_t5 = xr.open_dataset(
        f'{static_base_path}/land_cubic/land_cubic.static.tile5.nc')
    land_static_cubic = xr.concat([land_static_cubic_t3, land_static_cubic_t5], dim='grid_yt')
    land_static_cubic['grid_yt'] = np.arange(1, len(land_static_cubic.grid_yt)+1)
    land_static_cubic['geolon_t'] = xr.where(
        land_static_cubic['geolon_t'] > 180, 
        land_static_cubic['geolon_t']-360, 
        land_static_cubic['geolon_t'])
    
    region_mask = river_static.rv_basin == basin_id
    region_mask_cubic = rv_static_cubic.rv_basin == basin_id
    land_area = river_static.land_area
    cell_area = land_static.cell_area
    land_area_cubic = rv_static_cubic.land_area
    mrso_sat = land_static.mrsofc
    # plot_seasonal_cycle_for_region(
    #     exp_names, exp_name_labels, base_path, variables, start_year, end_year,
    #     region_mask, region_mask_cubic, region_label, mean_types, min_precip,
    #     land_area, land_area_cubic, cell_area, mrso_sat
    #     )
    # plot_ar_count_seasonal_cycle_for_region(
    #     exp_names, exp_name_labels, base_path, start_year, end_year,
    #     region_mask, region_label, min_precip=1)
    
    # plot_labelled_basin_map(
    #     rv_static_cubic.rv_basin, rv_static_cubic.land_area,
    #     land_static_cubic.geolon_t, land_static_cubic.geolat_t, 
    #     outlet_markers=True
    #     )
        
    plot_water_balance_seasonal_cycle_for_region(
        exp_names, exp_name_labels, base_path, 
        ['pr'], 
        ['evap_land', 'rv_d_h2o', 'FWS', 'LWS', 'FWSv', 'LWSv'], 
        start_year, end_year, region_mask, region_mask_cubic, region_label, 
        land_area, land_area_cubic, cell_area)

if __name__ == '__main__':
    _main()