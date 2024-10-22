import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import data_util as du
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from shapely import concave_hull


def create_basin_gdf(ref_data, dist_data, basin_ids, land_area, dT, min_basin_area=50000, region='global'):
    # Use data_util to select the region
    # ref_data = du.select_region(ref_data, region)
    # dist_data = du.select_region(dist_data, region)
    # basin_ids = du.select_region(basin_ids, region)
    
    # Create a GeoDataFrame from the basin IDs
    lon = ref_data.geolon_t.values
    lat = ref_data.geolat_t.values
    df = pd.DataFrame({
        'basin_id': basin_ids.values.flatten(),
        'precip_ctrl': ref_data.precip.values.flatten() * land_area.values.flatten(),
        'precip_dist': dist_data.precip.values.flatten() * land_area.values.flatten(),
        'ar_pr_ctrl': ref_data.ar_pr.values.flatten() * land_area.values.flatten(),
        'ar_pr_dist': dist_data.ar_pr.values.flatten() * land_area.values.flatten(),
        'lon': lon.flatten(),
        'lat': lat.flatten(),
        'land_area': land_area.values.flatten(),
    })
    
    # Group by basin ID and calculate sum of precipitation changes
    basin_sums = df.groupby('basin_id').agg({
        'precip_ctrl': 'sum',
        'precip_dist': 'sum',
        'ar_pr_ctrl': 'sum',
        'ar_pr_dist': 'sum',
        'land_area': 'sum',
    })
    # Apply scaling factors, kg / s --> kg m⁻² / year
    basin_sums['precip_ctrl'] = basin_sums.precip_ctrl / basin_sums.land_area * 86400 * 365.25
    basin_sums['precip_dist'] = basin_sums.precip_dist / basin_sums.land_area * 86400 * 365.25
    basin_sums['ar_pr_ctrl'] = basin_sums.ar_pr_ctrl / basin_sums.land_area * 86400 * 365.25
    basin_sums['ar_pr_dist'] = basin_sums.ar_pr_dist / basin_sums.land_area * 86400 * 365.25
    # Ctrl climate AR precipitation fraction
    basin_sums['ar_pr_ctrl_fraction'] = basin_sums.ar_pr_ctrl / basin_sums.precip_ctrl * 100
    basin_sums['ar_pr_dist_fraction'] = basin_sums.ar_pr_dist / basin_sums.precip_dist * 100
    
    # Calculate total precip change and AR precip change
    basin_sums['precip_change'] = (basin_sums.precip_dist - basin_sums.precip_ctrl) / dT
    basin_sums['ar_pr_change'] = (basin_sums.ar_pr_dist - basin_sums.ar_pr_ctrl) / dT
    basin_sums['rel_precip_change'] = basin_sums.precip_change / basin_sums.precip_ctrl * 100
    basin_sums['rel_ar_pr_change'] = basin_sums.ar_pr_change / basin_sums.ar_pr_ctrl * 100
    # Calculate the contribution of AR precip change to total precip change
    basin_sums['ar_contribution_to_precip_change'] = (
        basin_sums.ar_pr_change / basin_sums.precip_change) * 100
    basin_sums['change_in_ar_pr_fraction'] = (
        basin_sums.ar_pr_dist_fraction - basin_sums.ar_pr_ctrl_fraction) / dT

    # Create geometries for each basin
    geometries = create_basin_geometries(basin_sums, df, land_area, basin_ids, min_basin_area)
    # Create a GeoDataFrame with unique basin geometries
    basin_gdf = gpd.GeoDataFrame(
        basin_sums,
        geometry=geometries,
        crs=ccrs.PlateCarree()
    )
    # Remove any basins with invalid geometries
    basin_gdf = basin_gdf[basin_gdf.geometry.is_valid]
    
    return basin_gdf

def create_basin_geometries(basin_sums, df, land_area, basin_ids, min_basin_area):
        geometries = []
        for basin_id in basin_sums.index:
            basin_points = df[df['basin_id'] == basin_id]
            if len(basin_points) > 2:  # Need at least 3 points to create a polygon
                hull = basin_points[['lon', 'lat']].values
                geometry = MultiPoint(hull).buffer(
                    0.75, cap_style='square', join_style='mitre').buffer(
                        -0.6, cap_style='square', join_style='mitre')
                
                # Check if the basin area is greater than or equal to 50000 km^2
                if land_area.where(basin_ids == basin_id).sum() >= min_basin_area*1e6:
                    geometries.append(geometry)
                else:
                    geometries.append(None)
            else:
                geometries.append(None)
        return geometries

def plot_basin_map_changes_with_warming(basin_gdf, start_year=None, end_year=None, region='global'):
    """
    Plot a figure with 4 subplots showing various precipitation changes.

    Args:
        basin_gdf (GeoDataFrame): GeoDataFrame containing basin data
        start_year (int): Start year of the data
        end_year (int): End year of the data
        region (str): Region name for the plot title

    Returns:
        tuple: Figure and axes objects
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 15),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    variables = ['rel_precip_change', 'rel_ar_pr_change',
                 'change_in_ar_pr_fraction',
                 'ar_contribution_to_precip_change']
    titles = ['Precip Change with Warming',
              'AR Precip Change with Warming',
              '$\Delta$(AR Precip / Precip)',
              '$\Delta$(AR Precip) / $\Delta$(Precip)']
    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']
    units = ['% K⁻¹', '% K⁻¹', '% K⁻¹', '% K⁻¹']
    vmin_vmax = [(-8, 8), (-8, 8), (-3, 3), (-100, 100)]
    for i, (variable, title, cmap, vmin_vmax, unit) in enumerate(zip(variables, titles, cmaps, vmin_vmax, units)):
        ax = axs[i]
        
        # Set vmin and vmax symmetrically around 0
        vmin, vmax = vmin_vmax

        im = basin_gdf.plot(column=variable, ax=ax,
                            legend=False, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im.get_children()[0], ax=ax, orientation='horizontal',
                            pad=0.08, aspect=30, shrink=0.6)
        cbar.set_label(f'{variable.replace("_", " ").title()} ({unit})')
        basin_gdf.boundary.plot(ax=ax, color='black')

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.LAKES)
        ax.set_extent([-125, -60, 25, 50], crs=ccrs.PlateCarree())

        # ax.gridlines(draw_labels=True)

        ax.set_title(f'{title}\n{start_year}-{end_year}, {region.upper()}')

    plt.tight_layout()
    return fig, axs

def plot_basin_map_ctrl_climate(basin_gdf, start_year=None, end_year=None, region='global'):
    """
    Plot a figure with 3 subplots showing control climate precipitation variables.

    Args:
        basin_gdf (GeoDataFrame): GeoDataFrame containing basin data
        start_year (int): Start year of the data
        end_year (int): End year of the data
        region (str): Region name for the plot title

    Returns:
        tuple: Figure and axes objects
    """
    fig, axs = plt.subplots(1, 3, figsize=(24, 8),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    variables = ['precip_ctrl', 'ar_pr_ctrl', 'ar_pr_ctrl_fraction']
    titles = ['Total Precipitation', 'AR Precipitation', 'AR Precipitation Fraction']
    cmaps = ['viridis', 'viridis', 'YlOrRd']
    vmin_vmax = [(0, 1500), (0, 1500), (0, 75)]
    units = ['kg m⁻² year⁻¹', 'kg m⁻² year⁻¹', '%']
    for i, (variable, title, cmap, vmin_vmax, unit) in enumerate(
        zip(variables, titles, cmaps, vmin_vmax, units)):
        ax = axs[i]
        
        vmin, vmax = vmin_vmax

        im = basin_gdf.plot(column=variable, ax=ax,
                            legend=False, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im.get_children()[0], ax=ax, orientation='horizontal',
                            pad=0.08, aspect=30, shrink=0.6)
        cbar.set_label(f'{variable.replace("_", " ").title()} ({unit})')
        basin_gdf.boundary.plot(ax=ax, color='black')

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.LAKES)
        ax.set_extent([-125, -60, 25, 50], crs=ccrs.PlateCarree())

        # ax.gridlines(draw_labels=True)

        ax.set_title(f'{title} ({unit})\n{start_year}-{end_year}, {region.upper()}')

    plt.tight_layout()
    return fig, axs

def get_base_paths():
    """
    Returns the experiment base path map.

    Returns:
    - dict: Experiment base path map
    """
    return {
        'c192L33_am4p0_amip_HIRESMIP_HX': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_HX_p2K': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020': '/archive/Ming.Zhao/awg/2022.03/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_p2K': '/archive/Ming.Zhao/awg/2023.04/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min': '/archive/Ming.Zhao/awg/2023.04/',
        'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K': '/archive/Ming.Zhao/awg/2023.04/',
        'c192_obs': '/archive/Ming.Zhao/awg/2022.03/'
    }

def get_base_path(exp_name, var):
    """
    Get the base path for a given experiment and variable.

    Parameters:
    - exp_name (str): Name of the experiment
    - var (str): Variable name

    Returns:
    - str: Base path for the given experiment and variable
    """
    missing_vars_path = '/archive/Marc.Prange/ts_all_missing_vars/'
    missing_vars = ['ar_pr', 'ar_pr_intensity', 'ar_pr_frequency', 'pr_intensity', 'pr_frequency']
    exp_basepath_map = get_base_paths()

    return (f"{missing_vars_path}{exp_name}/ts_all/" if var in missing_vars
            else f"{exp_basepath_map.get(exp_name, '')}{exp_name}/ts_all/")

def lon360_to_lon180(ds):
    """
    Convert longitude from 0-360 range to -180 to 180 range.

    Parameters:
    - ds (xarray.Dataset): Dataset with longitude coordinates

    Returns:
    - xarray.Dataset: Dataset with converted longitude coordinates
    """
    lon = ds.geolon_t
    lon_adj = xr.where(lon > 180, lon - 360, lon)
    ds = ds.assign_coords(geolon_t=lon_adj)
    return ds

def load_model_data(exp_name, variables, start_year, end_year, tiles=None):
    """
    Load model data for a given experiment name from ts_all directories.

    Parameters:
    - exp_name (str): Name of the experiment
    - variables (list): List of variable names to load
    - start_year (int): Start year of the data
    - end_year (int): End year of the data
    - tiles (list): Optional list of tile identifiers to load tiled data

    Returns:
    - xarray.Dataset: Dataset containing the loaded variables
    """
    if tiles:
        base_path = '/archive/Marc.Prange/ts_all_missing_vars/'
        subset = {
            'ar_pr': 'atmos_cmip', 'ar_pr_intensity': 'atmos_cmip', 
            'ar_pr_frequency': 'atmos_cmip', 'precip': 'land'}
        datasets = [xr.open_mfdataset(
            [f"{base_path}{exp_name}/ts_all/{subset[var]}.195101-202012.{var}.{tile}.nc" 
             for tile in tiles], 
            combine='nested', concat_dim='grid_yt')
                    .sel(time=slice(f"{start_year}", f"{end_year}"))
                    for var in variables]
        combined_ds = xr.merge(datasets)
        combined_ds['geolon_t'] = combined_ds.geolon_t.where(combined_ds.geolon_t != -1e20)
        combined_ds['geolat_t'] = combined_ds.geolat_t.where(combined_ds.geolat_t != -1e20)
        combined_ds['grid_yt'] = np.arange(1, len(combined_ds['grid_yt']) + 1)
        combined_ds = lon360_to_lon180(combined_ds)
    else:
        datasets = [xr.open_mfdataset(
            f"{get_base_path(exp_name, var)}{subset[var]}.{start_year}01-{end_year}12.{var}.nc", 
            combine='by_coords')
                    .sel(time=slice(f"{start_year}", f"{end_year}"))
                    for var in variables]
        combined_ds = xr.merge(datasets)
    return combined_ds

def load_static_river_data(exp_name, tiles):
    """
    Load static river data for a given experiment and multiple tiles.

    Parameters:
    - exp_name (str): Name of the experiment
    - tiles (list): List of tile identifiers (e.g., ['tile1', 'tile2', etc.])

    Returns:
    - xarray.Dataset: Combined dataset containing static river data for the specified tiles
    """
    exp_basepath_map = get_base_paths()
    base_path = exp_basepath_map.get(exp_name, '')
    
    datasets = []
    for tile in tiles:
        # Construct the file path for static river data
        file_path = f"{base_path}{exp_name}/gfdl*/pp/river_cubic/river_cubic.static.{tile}.nc"
        
        try:
            # Load the dataset
            ds = xr.open_mfdataset(file_path)
            datasets.append(ds)
        except FileNotFoundError:
            print(f"Static river data file not found for {exp_name}, {tile}")
        except Exception as e:
            print(f"Error loading static river data for {exp_name}, {tile}: {str(e)}")
    
    if not datasets:
        return None
    
    # Combine datasets along the 'grid_yt' dimension
    combined_ds = xr.concat(datasets, dim='grid_yt')
    combined_ds['grid_yt'] = np.arange(1, len(combined_ds['grid_yt']) + 1)
    
    return combined_ds

def main():
    # Define experiment names and parameters
    ctrl_exp = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min'
    dist_exp = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K'
    exp_label_ref = 'nudge_30min_ctrl'
    exp_label_dist = 'nudge_30min_p2K'
    start_year = 1951
    end_year = 2020
    variables = ['precip', 'ar_pr']
    tiles = ['tile3','tile5']

    # Load model data for both experiments
    ctrl_data = load_model_data(ctrl_exp, variables, start_year, end_year, tiles).load()
    dist_data = load_model_data(dist_exp, variables, start_year, end_year, tiles).load()
    ctrl_data_mean = ctrl_data.mean(dim='time')
    dist_data_mean = dist_data.mean(dim='time')
    dT = du.get_global_mean_dT(ctrl_exp, dist_exp, start_year, end_year)

    # Load static river data for the control experiment
    static_river_data = load_static_river_data(ctrl_exp, tiles).load()
    basin_ids = static_river_data.rv_basin
    land_area = static_river_data.land_area
    # Create a GeoDataFrame using the create_basin_gdf function
    basin_gdf = create_basin_gdf(ctrl_data_mean, dist_data_mean, basin_ids, land_area, dT)
    # Plot control climate precipitation variables
    fig, axs = plot_basin_map_ctrl_climate(basin_gdf, start_year=start_year, end_year=end_year, region='CONUS')
    fig.savefig(f'plots/clim_mean_warming_trend_plots/{exp_label_dist}-{exp_label_ref}/{exp_label_ref}_{exp_label_dist}_ctrl_climate_basin_scale.png')
    # Plot AR contribution to precip change with warming on river basin scale
    fig, axs = plot_basin_map_changes_with_warming(
        basin_gdf,start_year=start_year, end_year=end_year, region='CONUS')
    fig.savefig(f'plots/clim_mean_warming_trend_plots/{exp_label_dist}-{exp_label_ref}/{exp_label_ref}_{exp_label_dist}_warming_changes_basin_scale.png')

if __name__ == "__main__":
    main()
