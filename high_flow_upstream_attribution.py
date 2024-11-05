import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import MultiPoint, Polygon
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from datetime import datetime
import geopandas as gpd
import json
import argparse

from store_low_high_flow_thresholds import (
    calculate_flow_extreme,
    load_daily_data,
)
from plot_to_cell import find_basin, find_upstream_ij
from data_util import sel_conus_land, lon_180_to_360, load_model_data

def get_true_coordinates(dataarray: xr.DataArray) -> list:
    """
    Returns a list of (time, lat, lon) tuples where the DataArray is True.

    Parameters:
        dataarray (xr.DataArray): A 3D boolean DataArray with dimensions (time, lat, lon).

    Returns:
        list of tuples: Each tuple contains (time, lat, lon) where the DataArray is True.
    """
    # Ensure the DataArray is boolean
    if dataarray.dtype != bool:
        raise ValueError("DataArray must be of boolean type.")

    # Stack the DataArray to collapse dimensions into a single dimension
    stacked = dataarray.stack(all_dims=['time', 'lat', 'lon'])

    # Select only the True values
    true_stacked = stacked.where(stacked).dropna(dim='all_dims', how='all')

    return true_stacked.coords['all_dims'].values

def find_nearest_grid_point(lat, lon, geolat, geolon):
    """
    Finds the nearest grid point indices in the geolat and geolon grids to the given latitude and longitude.

    Parameters:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        geolat (xr.DataArray): Latitude grid.
        geolon (xr.DataArray): Longitude grid.

    Returns:
        tuple: (y_index, x_index) of the nearest grid point.
    """

    # Calculate the squared distance between the input point and all grid points
    distance_sq = (geolat.values - lat) ** 2 + (geolon.values - lon) ** 2

    # Find the index of the minimum distance
    flat_index = np.nanargmin(distance_sq)
    x_idx, y_idx = np.unravel_index(flat_index, geolat.shape)

    return x_idx, y_idx

def perform_lagged_correlation(
        precip, ar_precip, melt, streamflow, high_flow_time, lag_days, temporal_window=10,
        plot_lagged_timeseries=False, min_corr=0.8):
    """
    Performs lagged correlation analysis between precipitation/snow changes and streamflow within a temporal window.

    Parameters:
        precip (xr.DataArray): Time series of total upstream precipitation.
        melt (xr.DataArray): Time series of snow melt.
        streamflow (xr.DataArray): Time series of streamflow at the high flow point.
        high_flow_time (np.datetime64 or pandas.Timestamp): The time of the high flow event.
        max_lag_days (int): Maximum number of days to lag. Default is 10.
        temporal_window (int): Number of days around the high flow day to consider. Default is 10.
        plot_lagged_timeseries (bool): Whether to plot lagged time series. Default is False.
        min_corr (float): Minimum correlation threshold. Default is 0.8.

    Returns:
        tuple: (best_lag, best_corr, source) where:
            - best_lag (int): Optimal lag time in days
            - best_corr (float): Best correlation coefficient
            - source (str): Either 'snow' or 'precip' indicating which variable had better correlation
            Returns (None, None, None) if neither correlation exceeds min_corr threshold
    """
    # Ensure the time dimension is sorted
    precip = precip.sortby('time')
    ar_precip = ar_precip.sortby('time')
    streamflow = streamflow.sortby('time')
    melt = melt.sortby('time')
    # Initialize best correlation values
    best_lag_precip = None
    best_corr_precip = -np.inf
    best_corr_precip_window = None
    best_lag_ar_precip = None
    best_corr_ar_precip = -np.inf
    best_corr_ar_precip_window = None
    best_lag_melt = None
    best_corr_melt = np.inf
    best_corr_melt_window = None
    best_corr_streamflow_window = None

    # Only positive lags since precipitation precedes streamflow
    for lag in lag_days:
        # Shift precipitation forward in time by 'lag' days
        shifted_precip = precip.shift(time=lag)
        shifted_ar_precip = ar_precip.shift(time=lag)
        shifted_melt = melt.shift(time=lag)
        # Define the time window around the high_flow_time after shifting
        start_time = high_flow_time - np.timedelta64(int(temporal_window/2), 'D')
        end_time = high_flow_time + np.timedelta64(int(temporal_window/2), 'D')

        # Select the windowed data after shifting
        precip_window = shifted_precip.sel(time=slice(start_time, end_time))
        melt_window = shifted_melt.sel(time=slice(start_time, end_time))
        streamflow_window = streamflow.sel(time=slice(start_time, end_time))
        ar_precip_window = shifted_ar_precip.sel(time=slice(start_time, end_time))
        # Check if upstream precip and melt sum is greater than streamflow on high flow day
        # Minimum criterion for precip or melt to cause high streamflow
        if precip_window.sum() > streamflow_window.sum():
            corr_matrix_precip = np.corrcoef(precip_window.values, streamflow_window.values)
            corr_precip = corr_matrix_precip[0, 1]
            if corr_precip > best_corr_precip:
                best_corr_precip = corr_precip
                best_lag_precip = lag
                best_corr_precip_window = precip_window
                best_corr_streamflow_window = streamflow_window
        else:
            print(f"Not enough precipitation to cause high flow on day {high_flow_time}", flush=True)
            corr_precip = np.nan
        if ar_precip_window.sum() > streamflow_window.sum():
            corr_matrix_ar_precip = np.corrcoef(ar_precip_window.values, streamflow_window.values)
            corr_ar_precip = corr_matrix_ar_precip[0, 1]
            if corr_ar_precip > best_corr_ar_precip:
                best_corr_ar_precip = corr_ar_precip
                best_lag_ar_precip = lag
                best_corr_ar_precip_window = ar_precip_window
                best_corr_streamflow_window = streamflow_window
        else:
            print(f"Not enough AR precipitation to cause high flow on day {high_flow_time}", flush=True)
            corr_ar_precip = np.nan
        if np.abs(melt_window.sum()) > streamflow_window.sum():
            corr_matrix_melt = np.corrcoef(melt_window.values, streamflow_window.values)
            corr_melt = corr_matrix_melt[0, 1]
            if corr_melt < best_corr_melt:
                best_corr_melt = corr_melt
                best_lag_melt = lag
                best_corr_melt_window = melt_window
                best_corr_streamflow_window = streamflow_window
        else:
            print(f"Not enough melt to cause high flow on day {high_flow_time}", flush=True)
            corr_melt = np.nan
        if plot_lagged_timeseries:
            plot_precip_streamflow_window(
                high_flow_time, precip_window, ar_precip_window, melt_window, streamflow_window, 
                lag, corr_precip, corr_ar_precip, corr_melt, 
                save_path='plots/high_low_flow_stat_plots/'
                          f'attribution_tests/streamflow_precip_melt_correlation_lag{lag}.png')

    # Return best correlation and source, 
    # prioritizing AR precipitation, then precipitation, then melt
    if best_corr_ar_precip > min_corr:
        return ('ar_precip', best_corr_ar_precip, 
                best_lag_ar_precip, best_corr_ar_precip_window, best_corr_streamflow_window)
    elif best_corr_precip > min_corr:
        return ('precip', best_corr_precip, 
                best_lag_precip, best_corr_precip_window, best_corr_streamflow_window)
    elif best_corr_melt < -min_corr:
        return ('melt', best_corr_melt, 
                best_lag_melt, best_corr_melt_window, best_corr_streamflow_window)
    else:
        return (None, None, None, None, None)
    
def plot_precip_streamflow_window(
    high_flow_time, precip_window, ar_precip_window, melt_window, streamflow_window, lag, 
    corr_precip, corr_ar_precip, corr_melt, save_path
):
    """
    Plots the time series of precipitation, AR precipitation, snow melt and streamflow around the high flow event 
    with the applied lag. Annotates the plot with the correlation coefficients.

    Parameters:
        high_flow_time (np.datetime64 or pandas.Timestamp): The time of the high flow event.
        precip_window (xr.DataArray): Shifted precipitation time series within the window.
        ar_precip_window (xr.DataArray): Shifted AR precipitation time series within the window.
        melt_window (xr.DataArray): Shifted snow melt time series within the window.
        streamflow_window (xr.DataArray): Streamflow time series within the window.
        lag (int): The lag time in days applied to precipitation and melt.
        corr_precip (float): The correlation coefficient between precipitation and streamflow.
        corr_ar_precip (float): The correlation coefficient between AR precipitation and streamflow.
        corr_melt (float): The correlation coefficient between snow melt and streamflow.
        save_path (str): The file path where the plot will be saved.
    """
    fig = plt.figure(figsize=(12, 6))
    # Create twin axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot precipitation on left axis
    ax1.plot(precip_window['time'], precip_window.values, label='Upstream Precipitation', color='blue')
    ax1.plot(ar_precip_window['time'], ar_precip_window.values, label='Upstream AR Precipitation', color='lightblue')
    ax1.plot(melt_window['time'], melt_window.values, label='Upstream Snow Melt', color='blue', linestyle='--')
    ax1.set_ylabel('Precip + Snow Melt (kg day$^{-1}$)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot streamflow on right axis  
    ax2.plot(streamflow_window['time'], streamflow_window.values, label='Streamflow', color='green')
    ax2.set_ylabel('Streamflow (kg day$^{-1}$)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.axvline(x=high_flow_time, color='green', linestyle='--', label='High Flow Event')

    fig.suptitle(f'Precipitation, snow cover change, and streamflow time series\n'
              f'Lag: {lag} days, Corr Precip: {corr_precip:.2f}, Corr AR Precip: {corr_ar_precip:.2f}, Corr Melt: {corr_melt:.2f}')
    ax1.set_xlabel('Time')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}", flush=True)

def plot_upstream_area_for_high_flow_point(
        basin_lon_cubic, basin_lat_cubic, basin_lon_reg, basin_lat_reg, lon_high_flow_reg, 
        lat_high_flow_reg, lon_high_flow_cubic, lat_high_flow_cubic, basin_polygon_cubic, 
        basin_polygon, interior_basin_rings, 
        save_path='plots/high_low_flow_stat_plots/'
                 f'attribution_tests/upstream_basin_map.png'):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.scatter(basin_lon_cubic, basin_lat_cubic, marker='o', color='blue', label='upstream basin,\ncubic grid')
    ax1.scatter(basin_lon_reg, basin_lat_reg, marker='x', color='red', label='upstream basin,\nregular grid')
    ax1.scatter(lon_high_flow_reg, lat_high_flow_reg, marker='x', color='green', label='high flow point,\nregular grid')
    ax1.scatter(lon_high_flow_cubic, lat_high_flow_cubic, marker='o', color='green', label='high flow point,\ncubic grid')
    ax1.plot(*basin_polygon_cubic.exterior.xy, color='black', label='basin boundary', transform=ccrs.PlateCarree())
    for ring in interior_basin_rings:   
        ax1.plot(*ring, color='black', linestyle='-', transform=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.scatter(basin_lon_cubic, basin_lat_cubic, marker='o', color='blue', label='upstream basin,\ncubic grid')
    ax2.scatter(basin_lon_reg, basin_lat_reg, marker='x', color='red', label='upstream basin,\nregular grid')
    ax2.scatter(lon_high_flow_reg, lat_high_flow_reg, marker='x', color='green', label='high flow point,\nregular grid')
    ax2.scatter(lon_high_flow_cubic, lat_high_flow_cubic, marker='o', color='green', label='high flow point,\ncubic grid')
    ax2.plot(*basin_polygon.exterior.xy, color='black', label='basin boundary', transform=ccrs.PlateCarree())
    for ring in interior_basin_rings:
        ax2.plot(*ring, color='black', linestyle='-', transform=ccrs.PlateCarree())
    ax1.coastlines()
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Set region to CONUS
    ax2.set_extent([-125, -60, 25, 50])
    ax2.coastlines()
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Add rivers to both plots
    # ax1.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)
    # ax2.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')



def plot_streamflow_timeseries(streamflow_data: xr.DataArray, precip_data: xr.DataArray, 
                               high_flow_mask: xr.DataArray, high_flow_time: np.datetime64, 
                               location: tuple, save_path: str = None):
    """
    Plots the time series of daily streamflow for a given location, marking the identified high flow events.

    Parameters:
        streamflow_data (xr.DataArray): Time series of streamflow with dimensions (time, lat, lon).
        high_flow_mask (xr.DataArray): Boolean mask indicating high flow events with the same dimensions as streamflow_data.
        location (tuple): A tuple of (latitude, longitude) specifying the location to plot.
        save_path (str, optional): File path to save the plot. If None, the plot is displayed.

    Raises:
        ValueError: If the specified location is not found in the data.
    """
    lat, lon = location

    # Select the nearest grid point to the specified location
    try:
        streamflow_point = streamflow_data.sel(lat=lat, lon=lon, method='nearest')
        precip_point = precip_data.sel(lat=lat, lon=lon, method='nearest')
        high_flow_point = high_flow_mask.sel(lat=lat, lon=lon, method='nearest')
    except KeyError:
        raise ValueError(f"Location ({lat}, {lon}) not found in the data.")

    # Extract time series data
    time = streamflow_point['time'].values
    streamflow_values = streamflow_point.values
    precip_values = precip_point.values
    high_flow_events = streamflow_point['time'].values[high_flow_point.values]
    high_flow_value = streamflow_point.sel(time=high_flow_time).values

    # Create the plot
    plt.figure(figsize=(14, 7))
    # Create primary axis for streamflow
    ax1 = plt.gca()
    ax1.plot(time, streamflow_values, label='Daily Streamflow', color='blue')
    ax1.set_ylabel('Streamflow (kg day$^{-1}$)')
    
    # Create secondary axis for precipitation
    ax2 = ax1.twinx()
    ax2.plot(time, precip_values, label='Daily Precipitation', color='green')
    ax2.set_ylabel('Precipitation (kg m$^{-2}$ s$^{-1}$)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    # Mark high flow events
    ax1.scatter(high_flow_events, streamflow_values[high_flow_point.values], 
                color='gray', marker='o', label='High Flow Events')
    ax1.scatter(high_flow_time, high_flow_value, color='red', marker='o', label='High Flow Event')

    ax1.set_title(f'Daily Streamflow at Location (Lat: {lat}, Lon: {lon})')
    ax1.set_xlabel('Time')
    ax1.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Streamflow timeseries plot saved to {save_path}", flush=True)
    else:
        plt.show()

def calculate_upstream_variables(daily_data_reg, land_area, basin_lat_reg, basin_lon_reg, 
                                 lat_high_flow_reg, lon_high_flow_reg, precip_var, ar_precip_var, snow_var, 
                                 monthly_mean_flow_velocity, monthly_mean_streamflow):
    upstream_land_area = xr.concat([
        land_area.sel(lat=lat, lon=lon)
        for lat, lon in zip(basin_lat_reg, basin_lon_reg)
    ], dim='basin_point')
    # Upstream precipitation
    upstream_precip = xr.concat([
        daily_data_reg.sel(lat=lat, lon=lon)[precip_var] 
        for lat, lon in zip(basin_lat_reg, basin_lon_reg)
    ], dim='basin_point')
    # Upstream AR precipitation
    upstream_precip_sum = (upstream_precip * upstream_land_area).sum(dim='basin_point').load()*86400
    upstream_ar_precip = xr.concat([
        daily_data_reg.sel(lat=lat, lon=lon)[ar_precip_var] 
        for lat, lon in zip(basin_lat_reg, basin_lon_reg)
    ], dim='basin_point')
    upstream_ar_precip_sum = (upstream_ar_precip * upstream_land_area).sum(dim='basin_point').load()*86400
    # Upstream snow
    upstream_snow = xr.concat([
        daily_data_reg.sel(lat=lat, lon=lon)['prsn'] 
        for lat, lon in zip(basin_lat_reg, basin_lon_reg)
    ], dim='basin_point')
    upstream_snow_sum = (upstream_snow * upstream_land_area).sum(dim='basin_point').load()*86400
    upstream_snow_cover = xr.concat([
        daily_data_reg.sel(lat=lat, lon=lon)[snow_var] 
        for lat, lon in zip(basin_lat_reg, basin_lon_reg)
    ], dim='basin_point')
    upstream_snow_cover_sum = (upstream_snow_cover * upstream_land_area).sum(dim='basin_point').load()
    upstream_snow_cover_sum_change = upstream_snow_cover_sum.differentiate('time')*1e9*86400
    # Upstream melt
    upstream_melt_sum = upstream_snow_cover_sum_change - upstream_snow_sum
    # High streamflow
    streamflow = (daily_data_reg.sel(
            lat=lat_high_flow_reg, lon=lon_high_flow_reg)['rv_o_h2o'] * \
                land_area.sel(
                lat=lat_high_flow_reg, lon=lon_high_flow_reg)).load()*86400
    # Upstream mean streamflow
    monthly_mean_upstream_flow_velocity_mean = xr.concat([
        monthly_mean_flow_velocity.sel(lat=lat, lon=lon)
        for lat, lon in zip(basin_lat_reg, basin_lon_reg)
    ], dim='basin_point').mean(dim='basin_point')
    monthly_mean_streamflow = (monthly_mean_streamflow.sel(
            lat=lat_high_flow_reg, lon=lon_high_flow_reg) * \
                land_area.sel(
                lat=lat_high_flow_reg, lon=lon_high_flow_reg)).load()*86400
    return (upstream_precip_sum, upstream_ar_precip_sum, upstream_melt_sum, streamflow, 
            monthly_mean_upstream_flow_velocity_mean, monthly_mean_streamflow, 
            upstream_land_area.sum('basin_point'))

def store_correlation_results_to_geodataframe(results, output_filepath):
    """
    Stores the correlation analysis results into a GeoDataFrame and saves it to a file.

    Parameters:
        results (list of dict): List containing dictionaries with correlation analysis results for each high flow event.
        output_filepath (str): File path to save the GeoDataFrame (e.g., 'results/correlation_results.geojson').

    The GeoDataFrame includes the following columns:
        - high_flow_event (int): Identifier for the high flow event.
        - time (numpy.datetime64): Time of the high flow event.
        - lon_reg (float): Longitude on the regular grid.
        - lat_reg (float): Latitude on the regular grid.
        - lon_cubic (float): Longitude on the cubic grid.
        - lat_cubic (float): Latitude on the cubic grid.
        - upstream_basin_shape (Polygon): Shapely Polygon representing the upstream basin.
        - high_flow_streamflow (float): Streamflow on the high flow day.
        - upstream_length_scale (float): Calculated upstream length scale.
        - upstream_flow_velocity_mean (float): Mean upstream flow velocity.
        - scaled_upstream_flow_velocity_mean (float): Scaled upstream flow velocity mean.
        - streamflow_timescale (int): Streamflow timescale in days.
        - best_corr_lag_days (int, optional): Best lag days for correlation.
        - best_corr_var (str, optional): Variable with the best correlation ('precip', 'ar_precip', or 'melt').
        - best_corr_value (float, optional): Best correlation coefficient value.
        - best_corr_var_window_json (str, optional): JSON string of window values of best correlated variable.
        - streamflow_window_values_json (str, optional): JSON string of streamflow window values.
    """
    
    # Prepare data for GeoDataFrame
    data = []
    for event in results:
        event_data = {
            'high_flow_event': event['event'],
            'time': event['time'],
            'lon_reg': float(event['lon_reg']),
            'lat_reg': float(event['lat_reg']),
            'lon_cubic': float(event['lon_cubic']),
            'lat_cubic': float(event['lat_cubic']),
            'upstream_basin_shape': event['upstream_basin_shape'],
            'high_flow_streamflow': float(event['high_flow_streamflow']),
            'upstream_length_scale': float(event['upstream_length_scale']),
            'upstream_flow_velocity_mean': float(event['upstream_flow_velocity_mean']),
            'scaled_upstream_flow_velocity_mean': float(event['scaled_upstream_flow_velocity_mean']),
            'streamflow_timescale': int(event['streamflow_timescale'])
        }

        # Add correlation results if available
        if event['correlation']:
            corr = event['correlation']
            # Convert numpy arrays to lists and then to JSON strings
            best_corr_var_window = corr.get('best_corr_var_window_values')
            streamflow_window = corr.get('streamflow_window_values')
            
            event_data.update({
                'best_corr_lag_days': corr.get('best_lag_days'),
                'best_corr_var': corr.get('best_corr_var'),
                'best_corr_value': corr.get('best_corr_value'),
                'best_corr_var_window': json.dumps(best_corr_var_window.values.tolist()) if best_corr_var_window is not None else None,
                'streamflow_window': json.dumps(streamflow_window.values.tolist()) if streamflow_window is not None else None
            })
        else:
            event_data.update({
                'best_corr_lag_days': None,
                'best_corr_var': None,
                'best_corr_value': None,
                'best_corr_var_window_json': None,
                'streamflow_window_values_json': None
            })

        data.append(event_data)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry='upstream_basin_shape')

    # Set coordinate reference system (CRS) if known, e.g., WGS84
    gdf.set_crs(epsg=4326, inplace=True)

    # Save to file
    gdf.to_file(output_filepath, driver='GeoJSON')
    print(f"Correlation results saved to {output_filepath}", flush=True)

def read_high_flow_events(filepath):
    """
    Reads high flow events from a GeoJSON file, including any stored lists.

    Parameters:
        filepath (str): Path to the GeoJSON file containing high flow events

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the high flow events with parsed data
    """
    # Read the GeoJSON file with lists as strings
    gdf = gpd.read_file(filepath, converters={
        'best_corr_var_window': lambda x: np.array(json.loads(x)) if isinstance(x, str) else None,
        'streamflow_window': lambda x: np.array(json.loads(x)) if isinstance(x, str) else None
    })

    return gdf

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process high flow events for a given year and experiment')
    parser.add_argument('--year', type=int, help='Year to process', default=1980)
    parser.add_argument('--exp_name', type=str, help='Experiment name', default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract arguments
    year = args.year
    exp_name = args.exp_name
    base_path = '/archive/Ming.Zhao/awg/2023.04/'
    gfdl_processor = 'gfdl.ncrc5-intel23-classic-prod-openmp'
    flow_var = 'rv_o_h2o'
    precip_var = 'prli'
    ar_precip_var = 'ar_prli'
    snow_var = 'snw'
    variables = [flow_var, precip_var, snow_var, 'pr', 'prsn',]
    ar_condition = True
    min_ar_pr_threshold = 1/86400
    low_pr_value = 0
    min_pr_var = precip_var
    ar_masked_vars = [precip_var]
    do_test_plots = False

    # Load daily data
    daily_data_reg = lon_180_to_360(sel_conus_land(load_model_data(
        base_path, year, variables, exp_name, ar_condition, min_ar_pr_threshold, 
        low_pr_value, min_pr_var, ar_masked_vars, gfdl_processor, lon_180=True,
    ))).load()
    
    # Load flow threshold
    flow_threshold = xr.open_dataarray(
        f'/archive/Marc.Prange/discharge_statistics/'
        f'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/flow_1p0_year-1_threshold_'
        f'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_1980_2019.nc')
    flow_threshold = lon_180_to_360(sel_conus_land(flow_threshold))

    # Load clim mean flow velocities
    monthly_mean_flow_velocity_ds = xr.open_mfdataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/river/av/monthly_42yr/river.1979-2020.*.nc')['rv_veloc'].groupby('time.month').mean('time').load()
    monthly_mean_streamflow_ds = xr.open_mfdataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/river/av/monthly_42yr/river.1979-2020.*.nc')['rv_o_h2o'].groupby('time.month').mean('time').load()
    # Load static datasets
    river_static_t3 = xr.open_dataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/river_cubic/river_cubic.static.tile3.nc').load()
    river_static_t5 = xr.open_dataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/river_cubic/river_cubic.static.tile5.nc').load()
    land_static_t5 = xr.open_dataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/land_cubic/land_cubic.static.tile5.nc').load()
    land_static_t3 = xr.open_dataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/land_cubic/land_cubic.static.tile3.nc').load()
    land_static = xr.open_dataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/land/land.static.nc').load()
    river_static_cubic = xr.concat([river_static_t3, river_static_t5], dim='grid_yt').transpose('grid_xt', 'grid_yt')
    river_static_cubic['grid_yt'] = np.arange(1, river_static_cubic.grid_yt.size+1)
    land_static_cubic = xr.concat([land_static_t3, land_static_t5], dim='grid_yt').transpose('grid_xt', 'grid_yt')
    land_static_cubic['grid_yt'] = np.arange(1, land_static_cubic.grid_yt.size+1)
    geolat_cubic = land_static_cubic.geolat_t
    geolon_cubic = land_static_cubic.geolon_t
    river_dir_cubic = river_static_cubic.rv_dir
    land_area = land_static.land_area

    # Calculate high flow mask
    high_flow_mask = calculate_flow_extreme(daily_data_reg, flow_var, 'high', flow_threshold).load()
    
    # Get high flow event coordinates
    high_flow_time_lat_lon = get_true_coordinates(high_flow_mask)
    
    # Create MultiPoint object for regular grid points
    lon_lat_reg = np.vstack([(lon, lat) for lon in daily_data_reg.lon for lat in daily_data_reg.lat])
    reg_lon_lat_MP = MultiPoint(lon_lat_reg)
    # Loop over all high flow events
    results = []
    for i_high_flow, event_coords in enumerate(high_flow_time_lat_lon):
        time_high_flow = np.datetime64(event_coords[0])
        lat_high_flow_reg = event_coords[1]
        lon_high_flow_reg = event_coords[2]
        # Select month of high flow event from monthly mean flow velocity and streamflow
        monthly_mean_flow_velocity = monthly_mean_flow_velocity_ds.sel(month=event_coords[0].month)
        monthly_mean_streamflow = monthly_mean_streamflow_ds.sel(month=event_coords[0].month)
        # Select nearest grid point
        igrid, jgrid = find_nearest_grid_point(
            lat_high_flow_reg, lon_high_flow_reg, 
            geolat_cubic, geolon_cubic)
        lat_high_flow_cubic = geolat_cubic[igrid, jgrid].values
        lon_high_flow_cubic = geolon_cubic[igrid, jgrid].values
        print(f"Processing high flow event {i_high_flow+1}/{len(high_flow_time_lat_lon)}:", flush=True)
        print(f"Time: {time_high_flow}", flush=True)
        print(f"Regular Grid - Lat: {lat_high_flow_reg}, Lon: {lon_high_flow_reg}", flush=True)
        print(f"Cubic Grid - Lat: {lat_high_flow_cubic}, Lon: {lon_high_flow_cubic}", flush=True)
        
        # Find basin
        basin_ij = find_basin(river_dir_cubic, igrid, jgrid)
        basin_lat_cubic = np.array([geolat_cubic[i, j] 
                              for i, j in zip(basin_ij[:, 0], basin_ij[:, 1])])
        basin_lon_cubic = np.array([geolon_cubic[i, j] 
                              for i, j in zip(basin_ij[:, 0], basin_ij[:, 1])])
        print(f"Basin size: {len(basin_lat_cubic)} pixels.", flush=True)
        # if len(basin_lat_cubic) < 8:
        #     continue
        basin_lon_lat_cubic = np.vstack([basin_lon_cubic, basin_lat_cubic]).T
        basin_polygon_cubic = MultiPoint(basin_lon_lat_cubic).buffer(
                                0.55, cap_style='square', join_style='mitre').buffer(
                                -0.25, cap_style='square', join_style='mitre') 
        point_bool = np.array([basin_polygon_cubic.contains(point) 
                               for point in reg_lon_lat_MP.geoms])
        basin_lon_reg = lon_lat_reg[point_bool][:, 0]
        basin_lat_reg = lon_lat_reg[point_bool][:, 1]

        interior_basin_rings_cubic = [ring.xy for ring in basin_polygon_cubic.interiors]
        # Plot upstream basin map
        if do_test_plots:
            plot_upstream_area_for_high_flow_point(
                basin_lon_cubic, basin_lat_cubic, basin_lon_reg, basin_lat_reg, lon_high_flow_reg, 
                lat_high_flow_reg, lon_high_flow_cubic, lat_high_flow_cubic, basin_polygon_cubic, 
                basin_polygon_cubic, interior_basin_rings_cubic, 
                save_path=f'plots/high_low_flow_stat_plots/'
                        f'attribution_tests/upstream_basin_map.png')
            plot_streamflow_timeseries(
                streamflow_data=daily_data_reg[flow_var], precip_data=daily_data_reg[precip_var], 
                high_flow_mask=high_flow_mask, high_flow_time=time_high_flow,
                location=(lat_high_flow_reg, lon_high_flow_reg), 
                save_path=f'plots/high_low_flow_stat_plots/'
                        f'attribution_tests/annual_streamflow_timeseries.png')
        
        # Extract upstream precipitation, snow cover change and streamflow
        (upstream_precip_sum, upstream_ar_precip_sum, upstream_melt_sum, streamflow, 
         upstream_flow_velocity_mean, monthly_mean_streamflow, upstream_land_area_sum,) = \
            calculate_upstream_variables(
                daily_data_reg, land_area, basin_lat_reg, basin_lon_reg,
                lat_high_flow_reg, lon_high_flow_reg, precip_var, ar_precip_var, 
                snow_var, monthly_mean_flow_velocity, monthly_mean_streamflow)
        
        # Calculate length scale of upstream basin and streamflow timescale
        upstream_length_scale = np.sqrt(upstream_land_area_sum/np.pi) # m^2
        high_stream_flow = streamflow.sel(time=time_high_flow)
        scaled_upstream_flow_velocity_mean = upstream_flow_velocity_mean * \
            (high_stream_flow/monthly_mean_streamflow)
        streamflow_timescale = int(np.ceil(
            upstream_length_scale/scaled_upstream_flow_velocity_mean/86400)) # days
        lag_days = np.arange(streamflow_timescale - 1, streamflow_timescale + 2)
        # Perform lagged correlation
        correlation = perform_lagged_correlation(
            upstream_precip_sum, upstream_ar_precip_sum, upstream_melt_sum, streamflow, time_high_flow, 
            lag_days=lag_days, temporal_window=streamflow_timescale, 
            plot_lagged_timeseries=do_test_plots)
        # Print report of correlation results
        print(f"Correlation results for high flow event {i_high_flow+1}/{len(high_flow_time_lat_lon)}:", flush=True)
        if correlation[2] == 'ar_precip':
            print(f"Best correlation with AR precipitation: {correlation[1]:.2f} at lag {correlation[0]} days", flush=True)
        elif correlation[2] == 'precip':
            print(f"Best correlation with precipitation: {correlation[1]:.2f} at lag {correlation[0]} days", flush=True)
        elif correlation[2] == 'melt':
            print(f"Best correlation with melt: {correlation[1]:.2f} at lag {correlation[0]} days", flush=True)
        else: 
            print("No significant correlation found.", flush=True)
        # Collect results
        event_result = {
            'event': i_high_flow + 1,
            'time': time_high_flow,
            'lon_reg': lon_high_flow_reg,
            'lat_reg': lat_high_flow_reg,
            'lon_cubic': lon_high_flow_cubic,
            'lat_cubic': lat_high_flow_cubic,
            'upstream_basin_shape': basin_polygon_cubic,
            'high_flow_streamflow': streamflow.sel(time=time_high_flow).values,
            'upstream_length_scale': upstream_length_scale,
            'upstream_flow_velocity_mean': upstream_flow_velocity_mean.values,
            'scaled_upstream_flow_velocity_mean': scaled_upstream_flow_velocity_mean.values,
            'streamflow_timescale': streamflow_timescale,
            'correlation': None
        }
        if correlation[0]:
            event_result['correlation'] = {
                'best_corr_value': correlation[1],
                'best_corr_var': correlation[0],
                'best_lag_days': correlation[2],
                'best_corr_var_window_values': correlation[3], 
                'streamflow_window_values': correlation[4], 
            }
        results.append(event_result)

    store_correlation_results_to_geodataframe(
        results, 
        output_filepath=f'/archive/m2p/high_flow_event_attribution/'
                        f'{exp_name}/correlation_results_{year}.geojson'
    )
if __name__ == "__main__":
    main()