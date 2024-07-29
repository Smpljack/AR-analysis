import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping, Polygon, Point
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely import overlaps, contains, contains_properly
import glob


import data_util as du

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two (lat, lon) points
    """
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    return distance

def find_nearest_point(lat_grid, lon_grid, lat_point, lon_point):
    """
    Find the nearest point on a latitude/longitude grid to a given latitude/longitude point.

    Parameters:
        lat_grid (numpy.ndarray): 1D array of latitude values on the grid.
        lon_grid (numpy.ndarray): 1D array of longitude values on the grid.
        lat_point (float): Latitude of the given point.
        lon_point (float): Longitude of the given point.

    Returns:
        nearest_lat (float): Latitude of the nearest point on the grid.
        nearest_lon (float): Longitude of the nearest point on the grid.
    """
    # Calculate distances for latitude and longitude separately
    lat_distances = np.abs(lat_grid - lat_point)
    lon_distances = np.abs(lon_grid - lon_point)
    
    # Find the nearest latitude and longitude points
    nearest_lat_index = np.nanargmin(lat_distances)
    nearest_lon_index = np.nanargmin(lon_distances)
    
    # Get the nearest latitude and longitude
    nearest_lat = lat_grid[nearest_lat_index]
    nearest_lon = lon_grid[nearest_lon_index]
    return nearest_lat, nearest_lon

def load_grdc_data(subregion, base_path='/archive/Marc.Prange/grdc_discharge_data/'):
    """
    Load GRDC gauge dataset for given subregion.

    Args:
        subregion (str): Subregion name.
        base_path (str, optional): 
        Path to GRDC directory. Defaults to '/archive/Marc.Prange/grdc_discharge_data/'.

    Returns:
        (xr.Dataset, gpd.DataFrame): 
        Dataset containing GRDC gauge timeseries for subregion and associated region DataArray.
    """
    grdc = xr.open_dataset(f'{base_path}{subregion}/GRDC-Daily.nc')
    grdc['dsch'] = (grdc.runoff_mean * 1000 / grdc.area / 1e6)
    region = gpd.read_file(f'{base_path}{subregion}/stationbasins.geojson')
    grdc = grdc.isel(id=np.isin(grdc.id, region.grdc_no))
    region = region.loc[np.isin(region.grdc_no, grdc.id)]
    return grdc, region

def constrain_grdc_data_to_5sec_calibrated(
    grdc_data, grdc_region, update_catchment_shape=True,
    path_to_eval_data='/archive/Marc.Prange/grdc_evaluation_data_burek_2023/'):
    # Load grdc evaluation files
    grdc_eval_5sec = pd.read_csv(
        path_to_eval_data+'GRDC_station_select_for_5min.csv', usecols=np.arange(18, 32), 
        header=2, skipfooter=6414-3916, index_col=0)
    # Filter grdc stations to be used for 5sec resolution calibration
    grdc_region_use_for_cal = np.isin(grdc_region.grdc_no, grdc_eval_5sec.index)
    grdc_data_use_for_cal = np.isin(grdc_data.id, grdc_eval_5sec.index)
    grdc_region_for_cal = grdc_region.loc[grdc_region_use_for_cal]
    grdc_data_for_cal = grdc_data.isel(id=grdc_data_use_for_cal)
    # Assign corrected variables to grdc data/region objects
    for grdc_no in grdc_data_for_cal.id.values:
        for region_var, grdc_data_var, eval_var in zip(
                                            ['area_calc', 'lat_org', 'long_org'], 
                                            ['area', 'geo_y', 'geo_x'], 
                                            ['area5min.1', 'lat5min.1', 'lon5min.1']):
            # Assign corrected value to grdc region object
            grdc_region_for_cal.loc[
                np.isin(grdc_region_for_cal.grdc_no, grdc_no), 
                region_var] = grdc_eval_5sec.loc[grdc_no][eval_var]
            # Assign corrected value to grdc data object
            grdc_data_for_cal[grdc_data_var][
                np.isin(grdc_data_for_cal.id, grdc_no)] = grdc_eval_5sec.loc[grdc_no][eval_var]
            if update_catchment_shape:
                shape_for_cal = gpd.read_file(
                    path_to_eval_data+ f'ashape5min/grdc_basin_5min_basin_{int(grdc_no)}.shp')
                grdc_region_for_cal.loc[
                    np.isin(grdc_region_for_cal.grdc_no, grdc_no), 
                    'geometry'] = shape_for_cal.loc[0].values[0]
                
    return grdc_data_for_cal, grdc_region_for_cal

def interp_model_data_to_gauges(model_data, grdc_region, variables=['rv_o_h2o', 'rv_d_h2o', 'pr'], interp_method='linear'):
    """
    Interpolate model data to gauge locations

    Args:
        model_data (xr.Dataset): Model dataset containing variables.
        grdc_region (geopandas Dataframe): GRDC region dataset containing metadata describing catchments.
        variables (list, optional): List of model variables. Defaults to ['rv_o_h2o', 'rv_d_h2o', 'pr'].

    Returns:
        xr.Dataset: Model
    """
    model_gauge_dict = {
        var: (('id', 'time'), 
              np.array([model_data[var].interp(lat=lat, lon=lon, method=interp_method) 
                    for lat, lon in zip(grdc_region.lat_org, grdc_region.long_org)]))
        for var in variables}
    model_gauge_dict['lat'] = (('id'), [d.lat for d in model_gauge_dsch])
    model_gauge_dict['lon'] = (('id'), [d.lon for d in model_gauge_dsch])
    model_gauge_dsch = xr.Dataset(
        data_vars=model_gauge_dict, 
        coords={'id': grdc_region.grdc_no, 'time': model_data.time.values})
    return model_gauge_dsch

def model_basin_mask(model_data, region_geometry):
    """
    Return a model_data mask for a given geometry describing a region.

    Args:
        model_data (xr.Dataset): model dataset with lat/lon coordinates.
        region_geometry (shapely.Geometry): Geometry object describing a geographical region.

    Returns:
        xr.DataArray: Mask array with lat/lon coordinates.
    """
    return xr.DataArray(
        data=np.array([[region_geometry.contains(Point(lon, lat)) 
                        for lon in model_data.lon] 
                    for lat in model_data.lat]).squeeze(),
        coords=(model_data.lat, model_data.lon)
        )

def interp_model_point_to_gauge(model_data, gauge_lat, gauge_lon, method='nearest'):
    return model_data.interp({'lon': gauge_lon, 'lat': gauge_lat}, method='nearest')
    
def get_closest_cama_point_to_gauge(cama_data, grdc_alloc):
    # Select lat/lon index, accounting for 0 indexing in Python
    cama_grdc_lat_ind = list(grdc_alloc.iy1.values-1)
    cama_grdc_lon_ind = list(grdc_alloc.ix1.values-1)
    if grdc_alloc.ix2.values[0] != -999:
        cama_grdc_lat_ind.append(grdc_alloc.iy2.values-1)
        cama_grdc_lon_ind.append(grdc_alloc.ix2.values-1)
    cama_data_closest = cama_data.isel({'lat': cama_grdc_lat_ind, 'lon': cama_grdc_lon_ind})
    return cama_data_closest

def spatial_integral(data_array):
    """
    Compute the spatial integral of a variable over a lat/lon grid using xarray.

    Parameters:
    - data_array (xarray.DataArray): Input data array with latitude and longitude dimensions.

    Returns:
    - integrated_variable (xarray.DataArray): Spatially integrated variable.
    """
    # Assuming latitude is 'lat' and longitude is 'lon'
    lat_rad = np.radians(data_array['lat'])
    lon_rad = np.radians(data_array['lon'])
    
    # Calculate grid cell areas
    # Assuming the Earth is a perfect sphere with radius 6371 km
    earth_radius_km = 6371
    delta_lat = np.gradient(lat_rad)
    delta_lon = np.gradient(lon_rad)
    grid_cell_areas = earth_radius_km**2 * np.outer(delta_lat, delta_lon) * np.cos(lat_rad)
    
    # Perform the integration
    integrated_variable = (data_array * grid_cell_areas).sum(dim=['lat', 'lon'])
    
    return integrated_variable

def multiply_vectors(vec1, vec2):
    # Reshape the vectors to have one dimension with size 1 to enable broadcasting
    vec1_reshaped = vec1[:, np.newaxis]
    vec2_reshaped = vec2[np.newaxis, :]

    # Perform element-wise multiplication
    result = vec1_reshaped * vec2_reshaped

    return result

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in meters
    R = 6371000  # Radius of Earth in meters
    return R * c

def get_half_lon_lat_grid_with_edges(longitudes, latitudes):
    """
    Get longitude/latitude grid on half-levels on input grid.
    Edges are added that are delta_x/2 and delta_y/2 out from
    the original coordinates.
    Args:
        longitudes (np.array): 1D array containing longitudes
        latitudes (np.array): 1D array containing latitudes

    Returns:
        tuple: tuple of two 1D arrays containing longitudes and latitudes
               on half levels, incl. added edges.
    """
    half_lons = np.concatenate(
        [[longitudes[0] - np.diff(longitudes)[0]/2], 
        (longitudes[1:] + longitudes[:-1])/2,
        [longitudes[-1] + np.diff(longitudes)[-1]/2]])
    half_lats = np.concatenate(
        [[latitudes[0] - np.diff(latitudes)[0]/2], 
        (latitudes[1:] + latitudes[:-1])/2,
        [latitudes[-1] + np.diff(latitudes)[-1]/2]])
    return half_lons, half_lats

def calculate_grid_area(longitudes, latitudes):
    """
    Calculate the area of each grid cell in square meters 
    given arrays of latitudes and longitudes (in degrees)
    """
    # Get grid on half-levels with added edges
    half_lons, half_lats = get_half_lon_lat_grid_with_edges(longitudes, latitudes)
    # Create mesh-grid
    half_lons_mesh, half_lats_mesh = np.meshgrid(half_lons, half_lats)
    # Calculate the distance between adjacent latitude points
    lat_distance = haversine(
        half_lons_mesh[1:, :], half_lats_mesh[:-1, :], 
        half_lons_mesh[1:, :], half_lats_mesh[1:, :])[:, 1:]
    
    # Calculate the distance between adjacent longitude points
    lon_distance = haversine(
        half_lons_mesh[:, :-1], half_lats_mesh[:, 1:], half_lons_mesh[:, 1:], half_lats_mesh[:, 1:])[1:, :]
    
    # Calculate the area of each grid cell
    area = lat_distance * lon_distance
    
    return area

def get_lin_reg_coefs(x, y): 
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def get_closest_cama_point_to_gauge(cama_data, grdc_alloc):
    # Select lat/lon index, accounting for 0 indexing in Python
    cama_grdc_lat_ind = list(grdc_alloc.iy1.values-1)
    cama_grdc_lon_ind = list(grdc_alloc.ix1.values-1)
    if grdc_alloc.ix2.values[0] != -999:
        cama_grdc_lat_ind.append(grdc_alloc.iy2.values-1)
        cama_grdc_lon_ind.append(grdc_alloc.ix2.values-1)
    cama_data_closest = cama_data.isel({'lat': cama_grdc_lat_ind, 'lon': cama_grdc_lon_ind})
    return cama_data_closest

def plot_region_map_ro_ratio_biases(
    region_df, grdc_ds, grdc_cama_alloc, model_pr, model_dsch, obs_pr,
    model_pr_to_daily=86400, obs_pr_to_daily=86400, model_dsch_to_daily=86400, obs_dsch_to_daily=86400):

    model_dsch_reversed_lat = model_dsch.reindex(lat=list(reversed(model_dsch.lat)))
    fig = plt.figure(figsize=(20, 25))
    gs = gridspec.GridSpec(
        nrows=3, ncols=3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1], hspace=0.2, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax6 = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
    ax7 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    ax8 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
    ax9 = fig.add_subplot(gs[2, 2], projection=ccrs.PlateCarree())
    region4plot = region_df.sort_values('area', ascending=False).set_index('grdc_no')
    region4plot['pr_model'] = np.nan * np.ones((len(region4plot)))
    region4plot['pr_obs'] = np.nan * np.ones((len(region4plot)))
    region4plot['pr_diff_model_obs'] = np.nan * np.ones((len(region4plot)))
    region4plot['dc_model'] = np.nan * np.ones((len(region4plot)))
    region4plot['dc_obs'] = np.nan * np.ones((len(region4plot)))
    region4plot['dc_diff_model_obs'] = np.nan * np.ones((len(region4plot)))
    region4plot['ro_ratio_model'] = np.nan * np.ones((len(region4plot)))
    region4plot['ro_ratio_obs'] = np.nan * np.ones((len(region4plot)))
    region4plot['ro_ratio_diff_model_obs'] = np.nan * np.ones((len(region4plot)))

    basin_ext_xx_min = 360
    basin_ext_xx_max = -180
    basin_ext_yy_min = 360
    basin_ext_yy_max = -180

    for grdc_no, basin in list(region4plot.iterrows()):
        print(f"{grdc_no}\t{basin.river}")
        grdc_basin = grdc_ds.sel(id=int(grdc_no))
        basin_scale = np.sqrt(grdc_basin.area.values)
        if (basin_scale < 50):
            print(f"Basin scale too small (~{np.round(basin_scale, 2)} km) for comparison to model.")
            region4plot = region4plot.drop(grdc_no)
            continue
        # elif (basin_scale > 500): 
        #     print(f"Basin scale too large (~{np.round(basin_scale, 2)} km) for now...")
        #     continue
        basin_gauge_in_cama_alloc = np.isin(grdc_cama_alloc.index, int(grdc_no))
        if not basin_gauge_in_cama_alloc.any():
            print("Basin Gauge could not be allocated to CaMa-Flood pixel. Continuing...")
            region4plot = region4plot.drop(grdc_no)
            continue
        overlap_with_prior_basins = np.array([
            (contains(prior_basin_geometry, basin.geometry) & ~contains(basin.geometry, prior_basin_geometry) )
            for prior_basin_geometry in region4plot.loc[:grdc_no].geometry])
        if overlap_with_prior_basins.any():
            print(f"Overlapping with prior basin, skipping this one...")
            region4plot = region4plot.drop(grdc_no)
            continue
        basin_ext_xx, basin_ext_yy = basin.geometry.exterior.coords.xy
        basin_ext_xx = np.array(basin_ext_xx)
        basin_ext_yy = np.array(basin_ext_yy)
        basin_ext_xx_min = np.min([basin_ext_xx.min(), basin_ext_xx_min])
        basin_ext_xx_max = np.max([basin_ext_xx.max(), basin_ext_xx_max])
        basin_ext_yy_min = np.min([basin_ext_yy.min(), basin_ext_yy_min])
        basin_ext_yy_max = np.max([basin_ext_yy.max(), basin_ext_yy_max])
        model_basin_pr = model_pr.sel(
            {
                'lon': slice(basin_ext_xx.min()-1, basin_ext_xx.max()+1),
                'lat': slice(basin_ext_yy.min()-1, basin_ext_yy.max()+1)
            })
        model_basin_dsch = model_dsch_reversed_lat.sel(
            {
                'lon': slice(basin_ext_xx.min()-1, basin_ext_xx.max()+1),
                'lat': slice(basin_ext_yy.min()-1, basin_ext_yy.max()+1)
            })
        obs_basin_pr = obs_pr.sel(
            {
                'lon': slice(basin_ext_xx.min()-1, basin_ext_xx.max()+1),
                'lat': slice(basin_ext_yy.min()-1, basin_ext_yy.max()+1)
            })
        model_dsch_basin_mask = xr.DataArray(
            data=np.array([[basin.geometry.contains(Point(lon, lat)) 
                            for lon in model_basin_dsch.lon] 
                        for lat in model_basin_dsch.lat]).squeeze(),
            coords=(model_basin_dsch.lat, model_basin_dsch.lon)
            )
        
        # Interpolate precip to 10 km cama grid
        print("Interpolating precip data...")
        model_basin_pr = model_basin_pr.interp({'lat': model_basin_dsch.lat, 'lon': model_basin_dsch.lon})
        obs_basin_pr = obs_basin_pr.interp({'lat': model_basin_dsch.lat, 'lon': model_basin_dsch.lon})
        # Mask precip data to reduce to precip in basin
        model_basin_pr = model_basin_pr.where(model_dsch_basin_mask)
        obs_basin_pr = obs_basin_pr.where(model_dsch_basin_mask)

        # Take discharge at point closest to gauge (transform to kg s^-1)
        model_gauge_dsch = get_closest_cama_point_to_gauge(model_dsch, grdc_cama_alloc.loc[basin_gauge_in_cama_alloc])
        model_gauge_dsch = model_gauge_dsch.sum(['lat', 'lon'])*1000 # model discharge, to kg s^-1
        # Store coordinate of nearest point
        # nearest_model_lat, nearest_model_lon = find_nearest_point(
        #     model_gauge_dsch.lat.values, model_gauge_dsch.lon.values, basin.lat_org, basin.long_org)
        # dist_model_gauge = haversine(nearest_model_lon, nearest_model_lat, basin.long_org, basin.lat_org)
        obs_gauge_dsch = grdc_basin.runoff_mean*1000 # obs discharge from gauge, kg s^-1
        # Spatially integrate precipitation
        model_pr_pixel_area = calculate_grid_area(model_basin_pr.lon.values, model_basin_pr.lat.values)
        obs_pr_pixel_area = calculate_grid_area(obs_basin_pr.lon.values, obs_basin_pr.lat.values)
        model_basin_pr_sum = (model_pr_pixel_area * model_basin_pr).sum(['lat', 'lon'])
        obs_basin_pr_sum = (obs_pr_pixel_area * obs_basin_pr).sum(['lat', 'lon'])
        # Get annual integrals
        model_basin_pr_sum_annual = (model_basin_pr_sum*model_pr_to_daily).groupby('time.year').sum('time')
        obs_basin_pr_sum_annual = (obs_basin_pr_sum*obs_pr_to_daily).groupby('time.year').sum('time')
        model_gauge_dsch_annual = (model_gauge_dsch*model_dsch_to_daily).groupby('time.year').sum('time')
        obs_gauge_dsch_annual = (obs_gauge_dsch*obs_dsch_to_daily).groupby('time.year').sum('time')
        # Get long-term mean of integrals
        model_basin_pr_sum_annual_mean = model_basin_pr_sum_annual.mean('year')
        obs_basin_pr_sum_annual_mean = obs_basin_pr_sum_annual.mean('year')
        model_gauge_dsch_annual_mean = model_gauge_dsch_annual.mean('year')
        obs_gauge_dsch_annual_mean = obs_gauge_dsch_annual.mean('year')
        region4plot.loc[grdc_no, 'pr_model'] = model_basin_pr_sum_annual_mean / basin.area*1e-6
        region4plot.loc[grdc_no, 'pr_obs'] = obs_basin_pr_sum_annual_mean / basin.area*1e-6
        region4plot.loc[grdc_no, 'pr_diff_model_obs'] = (
            model_basin_pr_sum_annual_mean - obs_basin_pr_sum_annual_mean) / obs_basin_pr_sum_annual_mean
        region4plot.loc[grdc_no, 'dc_model'] = model_gauge_dsch_annual_mean / basin.area*1e-6
        region4plot.loc[grdc_no, 'dc_obs'] = obs_gauge_dsch_annual_mean / basin.area*1e-6
        region4plot.loc[grdc_no, 'dc_diff_model_obs'] = (
            model_gauge_dsch_annual_mean - obs_gauge_dsch_annual_mean) / obs_gauge_dsch_annual_mean
        region4plot.loc[grdc_no, 'ro_ratio_model'] = model_gauge_dsch_annual_mean / model_basin_pr_sum_annual_mean
        region4plot.loc[grdc_no, 'ro_ratio_obs'] = obs_gauge_dsch_annual_mean / obs_basin_pr_sum_annual_mean
        region4plot.loc[grdc_no, 'ro_ratio_diff_model_obs'] = model_gauge_dsch_annual_mean / model_basin_pr_sum_annual_mean - \
            obs_gauge_dsch_annual_mean / obs_basin_pr_sum_annual_mean

    caxs = [make_axes_locatable(ax).append_axes('bottom', size='5%', pad=0.05, axes_class=maxes.Axes) 
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]]

    ax1 = region4plot.plot(column='ro_ratio_model', ax=ax1, cmap='viridis', 
                    vmin=0, vmax=1, cax=caxs[0], legend=True,
                    legend_kwds={
                        "label": "D/P model", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax2 = region4plot.plot(column='ro_ratio_obs', ax=ax2, cmap='viridis', 
                    vmin=0, vmax=1, cax=caxs[1], legend=True,
                    legend_kwds={
                        "label": "D/P obs@50km", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax3 = region4plot.plot(column='ro_ratio_diff_model_obs', ax=ax3, cmap='coolwarm', 
                    vmin=-0.5, vmax=0.5, cax=caxs[2], legend=True,
                    legend_kwds={
                        "label": "D/P (model-obs@50km)", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax4 = region4plot.plot(column='pr_model', ax=ax4, cmap='viridis', 
                    cax=caxs[3], legend=True,
                    legend_kwds={
                        "label": "Pr model / kg m${-2}$ year$^{-1}$", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax5 = region4plot.plot(column='pr_obs', ax=ax5, cmap='viridis', 
                    cax=caxs[4], legend=True,
                    legend_kwds={
                        "label": "Pr obs / kg m${-2}$ year$^{-1}$", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax6 = region4plot.plot(column='pr_diff_model_obs', ax=ax6, cmap='coolwarm', 
                    cax=caxs[5], legend=True, vmin=-1, vmax=1,
                    legend_kwds={
                        "label": "Pr diff (model-obs)/obs / -", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax7 = region4plot.plot(column='dc_model', ax=ax7, cmap='viridis', 
                    cax=caxs[6], legend=True,
                    legend_kwds={
                        "label": "Discharge model / kg m${-2}$ year$^{-1}$", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax8 = region4plot.plot(column='dc_obs', ax=ax8, cmap='viridis', 
                    cax=caxs[7], legend=True,
                    legend_kwds={
                        "label": "Discharge obs / kg m${-2}$ year$^{-1}$", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax9 = region4plot.plot(column='dc_diff_model_obs', ax=ax9, cmap='coolwarm', 
                    cax=caxs[8], legend=True, vmin=-1, vmax=1,
                    legend_kwds={
                        "label": "Discharge diff (model-obs)/obs / -", 
                        "orientation": "horizontal"},
                    edgecolor='black')
    ax1 = region4plot.plot(ax=ax1, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax2 = region4plot.plot(ax=ax2, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax3 = region4plot.plot(ax=ax3, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax4 = region4plot.plot(ax=ax4, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax5 = region4plot.plot(ax=ax5, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax6 = region4plot.plot(ax=ax6, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax7 = region4plot.plot(ax=ax7, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax8 = region4plot.plot(ax=ax8, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    ax9 = region4plot.plot(ax=ax9, kind="scatter", x="long_org", y="lat_org", marker='x', color='red')
    # Annotate each basin with D/P value
    region4plot['coords'] = region4plot['geometry'].apply(lambda x: x.representative_point().coords[:])
    region4plot['coords'] = [coords[0] for coords in region4plot['coords']]
    for idx, row in region4plot.iterrows():
        ax1.annotate(text=np.round(row['ro_ratio_model'], 2), xy=row['coords'],
                    horizontalalignment='center', color='white', fontsize=8)
        ax2.annotate(text=np.round(row['ro_ratio_obs'], 2), xy=row['coords'],
                    horizontalalignment='center', color='white', fontsize=8)
        ax3.annotate(text=np.round(row['ro_ratio_diff_model_obs'], 2), xy=row['coords'],
                    horizontalalignment='center', color='black', fontsize=8)
        ax6.annotate(text=np.round(row['pr_diff_model_obs'], 2), xy=row['coords'],
                    horizontalalignment='center', color='black', fontsize=8)
        ax9.annotate(text=np.round(row['dc_diff_model_obs'], 2), xy=row['coords'],
                    horizontalalignment='center', color='black', fontsize=8)

    edge = 1
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.coastlines("10m", linewidth=1)
        # axis.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='tab:blue', edgecolor='tab:blue', linewidth=0.5)
        ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='tab:blue', linewidth=1)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='black', linewidth=0.5)
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
        ax.set_extent(
            [basin_ext_xx_min-edge, basin_ext_xx_max+edge, 
            basin_ext_yy_min-edge, basin_ext_yy_max+edge], 
            crs=ccrs.PlateCarree())
    plt.tight_layout()
    return fig

def _main():
    grdc_list = []
    region_list = []
    for region_name in [
        'COLUMBIA', 'GREAT_BASIN',  'NORTH_PACIFIC',  'SACRAMENTO',  'SAN_JOAQUIN',
        'KLAMATH', 'ROGUE', 'SALINAS', 'SKAGIT', 'SNAKE']:
        grdc, region = load_grdc_data(region_name)
        grdc_list.append(grdc)
        region_list.append(region)
    grdc = xr.concat(grdc_list, dim='id')
    region = pd.concat(region_list)

    # region_cmb_geometry = unary_union(region.geometry)
    grdc, region = constrain_grdc_data_to_5sec_calibrated(
        grdc, region
        )
    
    path_to_eval_data='/archive/Marc.Prange/cama-flood/'
    grdc_alloc = pd.read_csv(
        path_to_eval_data+'GRDC_alloc_5min_lat_lon_area_cal.txt',
        header=0, index_col=0, sep='\s+')
    print("Reading AM4/LM4.0 data...")
    paths = [p for p in 
    glob.glob('/archive/Marc.Prange/na_data/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
            'c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020_na_*.nc') 
    if int(p[-7:-3]) in range(1980, 2020)]
    model_data = xr.open_mfdataset(paths).load()
    model_river_static = du.sel_na(du.lon_360_to_180(xr.open_dataset(
        '/archive/Ming.Zhao/awg/2022.03/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/gfdl.ncrc4-intel-prod-openmp/pp/river/'
        'river.static.nc')))
    model_data['vol_discharge'] = (
        model_data.rv_o_h2o + model_data.rv_d_h2o) * model_river_static.land_area / 1000
    model_data['vol_discharge_o'] = (
        model_data.rv_o_h2o) * model_river_static.land_area / 1000
    model_data['vol_discharge_d'] = (
        model_data.rv_d_h2o) * model_river_static.land_area / 1000
    model_data['vol_runoff'] = model_data.mrro * model_river_static.land_area / 1000
    print("Reading obs precip data...") 
    paths = [p for p in 
    glob.glob('/archive/Marc.Prange/na_data/c192_obs/'
            'c192_obs_na_*.nc')
    if int(p[-7:-3]) in range(1980, 2020)]
    era5_data = xr.open_mfdataset(paths).load()
    print("Reading CaMa-Flood data...")
    paths = [p for p in
         glob.glob(
             '/archive/Marc.Prange/cama-flood/'
             'LM4-conus_06min_1951_2020/o_outflw*.nc')]
    cama_data = xr.open_mfdataset(paths).load()
    print("Reversing order of latitude coordinate...")
    cama_data_reverse_lat = cama_data.reindex(lat=list(reversed(cama_data.lat)))

    grdc = grdc.sel(time=slice('1980', '2007'))
    model_data = model_data.sel(time=slice('1980', '2007'))
    cama_data = cama_data.sel(time=slice('1980', '2007'))
    cama_data_reverse_lat = cama_data_reverse_lat.sel(time=slice('1980', '2007'))
    era5_data = era5_data.sel(time=slice('1980', '2007'))
    
    fig = plot_region_map_ro_ratio_biases(
        region, grdc, grdc_alloc, model_data.pr, cama_data.outflw, era5_data.pr,)
    plt.savefig('plots/test_catchment_map.png')
    
if __name__ == '__main__':
    _main()