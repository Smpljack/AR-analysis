import os
import json
import argparse
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import MultiPoint
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from store_low_high_flow_thresholds import calculate_flow_extreme
from data_util import sel_na_land, lon_180_to_360, load_model_data


def get_true_coordinates(dataarray: xr.DataArray) -> list:
    """
    Returns a list of (time, lat, lon) tuples where the DataArray is True.

    Parameters:
        dataarray (xr.DataArray): A 3D boolean DataArray with dimensions 
                                   (time, lat, lon).

    Returns:
        list of tuples: Each tuple contains (time, lat, lon) where the 
                        DataArray is True.
    """
    # Ensure the DataArray is boolean
    if dataarray.dtype != bool:
        raise ValueError("DataArray must be of boolean type.")

    # Stack the DataArray to collapse dimensions into a single dimension
    stacked = dataarray.stack(all_dims=['time', 'lat', 'lon'])

    # Select only the True values
    true_stacked = stacked.where(stacked).dropna(dim='all_dims', how='all')

    return true_stacked.coords['all_dims'].values


def perform_lagged_correlation(
        precip, ar_precip, melt, precip_var, ar_precip_var, melt_var, 
        streamflow, high_flow_time, lag_days, temporal_window=10, 
        plot_lagged_timeseries=False, min_corr=0.8):
    """
    Performs lagged correlation analysis between precipitation/snow changes 
    and streamflow within a temporal window.

    Parameters:
        precip (xr.DataArray): Time series of total upstream precipitation.
        melt (xr.DataArray): Time series of snow melt.
        streamflow (xr.DataArray): Time series of streamflow at the high 
                                    flow point.
        high_flow_time (np.datetime64 or pandas.Timestamp): The time of 
                                                            the high flow 
                                                            event.
        max_lag_days (int): Maximum number of days to lag. Default is 10.
        temporal_window (int): Number of days around the high flow day to 
                               consider. Default is 10.
        plot_lagged_timeseries (bool): Whether to plot lagged time series. 
                                        Default is False.
        min_corr (float): Minimum correlation threshold. Default is 0.8.

    Returns:
        tuple: (best_lag, best_corr, source) where:
            - best_lag (int): Optimal lag time in days
            - best_corr (float): Best correlation coefficient
            - source (str): Either 'snow' or 'precip' indicating which 
                            variable had better correlation
            Returns (None, None, None) if neither correlation exceeds 
            min_corr threshold
    """
    # Initialize best correlation values
    variables = ['ar_precip', 'precip', 'melt', 'ar_precip_melt', 
                 'precip_melt']  # Order of priority
    best_corr_values = {var: -np.inf for var in variables}
    best_corr_lag_values = {var: None for var in variables}
    best_corr_windows = {var: None for var in variables}
    best_corr_variances = {var: None for var in variables}
    best_corr_streamflow_windows = {var: None for var in variables}
    low_var_bool = {var: True for var in variables}  # Will be set to 
                                                      # False if upstream 
                                                      # sum is higher than 
                                                      # streamflow

    # Only positive lags since precipitation precedes streamflow
    for lag in lag_days:
        # Shift variables forward in time by 'lag' days
        shifted_vars = {
            'precip': precip.shift(time=lag),
            'ar_precip': ar_precip.shift(time=lag),
            'melt': melt.shift(time=lag)
        }
        shifted_vars['precip_variance'] = precip_var.shift(time=lag)
        shifted_vars['ar_precip_variance'] = ar_precip_var.shift(time=lag)
        shifted_vars['melt_variance'] = melt_var.shift(time=lag)

        # Define the time window around the high_flow_time after shifting
        start_time = high_flow_time - np.timedelta64(int(temporal_window), 
                                                      'D')
        end_time = high_flow_time + np.timedelta64(int(temporal_window), 
                                                    'D')

        # Select the windowed data after shifting
        windowed_vars = {var: shifted_vars[var].sel(
            time=slice(start_time, end_time)) for var in ['precip', 
                                                            'ar_precip', 
                                                            'melt']}
        windowed_vars['precip_variance'] = shifted_vars['precip_variance'].sel(
            time=high_flow_time)
        windowed_vars['ar_precip_variance'] = shifted_vars['ar_precip_variance'].sel(
            time=high_flow_time)
        windowed_vars['melt_variance'] = shifted_vars['melt_variance'].sel(
            time=high_flow_time)
        streamflow_window = streamflow.sel(time=slice(start_time, end_time))

        # Check correlations for each variable
        for var in ['precip', 'ar_precip', 'melt']:
            if windowed_vars[var].sum() > streamflow_window.sum():
                low_var_bool[var] = False
                corr_matrix = np.corrcoef(windowed_vars[var].values, 
                                           streamflow_window.values)
                corr_value = corr_matrix[0, 1]
                if corr_value > best_corr_values[var]:
                    best_corr_values[var] = corr_value
                    best_corr_lag_values[var] = lag
                    best_corr_windows[var] = windowed_vars[var]
                    best_corr_streamflow_windows[var] = streamflow_window
                    best_corr_variances[var] = windowed_vars[f'{var}_variance']

        # Check combined correlations
        for combined_var in ['precip_melt', 'ar_precip_melt']:
            combined_window = windowed_vars['precip'] + windowed_vars['melt'] \
                if combined_var == 'precip_melt' else windowed_vars['ar_precip'] + \
                windowed_vars['melt']
            if combined_window.sum() > streamflow_window.sum():
                low_var_bool[combined_var] = False
                corr_matrix = np.corrcoef(combined_window.values, 
                                           streamflow_window.values)
                corr_value = corr_matrix[0, 1]
                if corr_value > best_corr_values[combined_var]:
                    best_corr_values[combined_var] = corr_value
                    best_corr_lag_values[combined_var] = lag
                    best_corr_windows[combined_var] = combined_window
                    best_corr_streamflow_windows[combined_var] = streamflow_window
                    best_corr_variances[combined_var] = windowed_vars['precip_variance'] + \
                        windowed_vars['melt_variance'] \
                        if combined_var == 'precip_melt' else \
                        windowed_vars['ar_precip_variance'] + \
                        windowed_vars['melt_variance']

        if plot_lagged_timeseries:
            plot_precip_streamflow_window(
                high_flow_time, windowed_vars['precip'],
                windowed_vars['ar_precip'], windowed_vars['melt'],
                streamflow_window, lag, best_corr_values['precip'], 
                best_corr_values['ar_precip'], best_corr_values['melt'],
                save_path='plots/high_low_flow_stat_plots/'
                          f'attribution_tests/streamflow_precip_melt_correlation_lag{lag}.png')

    # Return best correlation and source,
    # prioritizing AR precipitation, then precipitation, then melt, 
    # then ar_precip+melt, then precip+melt
    for var in variables:
        if best_corr_values[var] > min_corr:
            best_corr_var = var
            break
        else:
            best_corr_var = 'no_corr'

    return {
        'best_corr_var': best_corr_var,
        'best_corr_value': {f'best_corr_value_{var}': best_corr_values[var] 
                            for var in variables},
        'best_corr_lag_days': {f'best_corr_lag_days_{var}': best_corr_lag_values[var] 
                                for var in variables},
        'best_corr_window_time': {f'best_corr_window_time_{var}': 
                                   best_corr_windows[var].time.values
                                   if best_corr_windows[var] is not None 
                                   else None for var in variables},
        'best_corr_window_var': {f'best_corr_window_{var}': 
                                 best_corr_windows[var].values
                                 if best_corr_windows[var] is not None 
                                 else None for var in variables},
        'best_corr_streamflow_window': {f'best_corr_streamflow_window_{var}': 
                                         best_corr_streamflow_windows[var].values
                                         if best_corr_streamflow_windows[var] 
                                         is not None else None for var in variables},
        'best_corr_variance': {f'best_corr_variance_{var}': 
                               best_corr_variances[var].values
                               if best_corr_variances[var] is not None 
                               else None for var in variables},
        'low_var_bool': {f'low_var_bool_{var}': low_var_bool[var] 
                         for var in variables}
    }


def plot_precip_streamflow_window(
        high_flow_time, precip_window, ar_precip_window, melt_window, 
        streamflow_window, lag, corr_precip, corr_ar_precip, corr_melt, 
        save_path):
    """
    Plots the time series of precipitation, AR precipitation, snow melt 
    and streamflow around the high flow event with the applied lag. 
    Annotates the plot with the correlation coefficients.

    Parameters:
        high_flow_time (np.datetime64 or pandas.Timestamp): The time of 
                                                            the high flow 
                                                            event.
        precip_window (xr.DataArray): Shifted precipitation time series 
                                       within the window.
        ar_precip_window (xr.DataArray): Shifted AR precipitation time 
                                          series within the window.
        melt_window (xr.DataArray): Shifted snow melt time series within 
                                     the window.
        streamflow_window (xr.DataArray): Streamflow time series within 
                                           the window.
        lag (int): The lag time in days applied to precipitation and melt.
        corr_precip (float): The correlation coefficient between 
                             precipitation and streamflow.
        corr_ar_precip (float): The correlation coefficient between AR 
                                precipitation and streamflow.
        corr_melt (float): The correlation coefficient between snow melt 
                           and streamflow.
        save_path (str): The file path where the plot will be saved.
    """
    fig = plt.figure(figsize=(12, 6))
    # Create twin axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot streamflow on right axis
    ax1.plot(streamflow_window['time'], streamflow_window.values, 
             label='Streamflow', color='green')
    ax1.set_ylabel('Streamflow (kg day$^{-1}$)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.scatter(high_flow_time, 
                streamflow_window.sel(time=high_flow_time).values, 
                marker='o', color='red', label='High Flow Event')

    # Plot precipitation on left axis
    ax2.plot(precip_window['time'], precip_window.values, 
             label='Upstream Precipitation', color='mediumblue')
    ax2.plot(ar_precip_window['time'], ar_precip_window.values, 
             label='Upstream AR Precipitation', color='cornflowerblue')
    ax2.plot(melt_window['time'], melt_window.values, 
             label='Upstream Snow Melt', color='purple')

    ax2.set_ylabel('Precip + Snow Melt (kg day$^{-1}$)', color='mediumblue')
    ax2.tick_params(axis='y', labelcolor='mediumblue')
    fig.suptitle(f'Precipitation, snow cover change, and streamflow time '
                  f'series\nLag: {lag} days, Corr Precip: {corr_precip:.2f}, '
                  f'Corr AR Precip: {corr_ar_precip:.2f}, Corr Melt: '
                  f'{corr_melt:.2f}')
    ax1.set_xlabel('Time')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}", flush=True)


def plot_upstream_area_for_high_flow_point(
        basin_lon_reg, basin_lat_reg, lon_high_flow, lat_high_flow, 
        basin_polygon, save_path='plots/high_low_flow_stat_plots/'
                  f'attribution_tests/upstream_basin_map.png'):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.plot(*basin_polygon.exterior.xy, color='black', 
             label='basin boundary', transform=ccrs.PlateCarree(), 
             linewidth=2)
    interior_basin_rings = [ring.xy for ring in basin_polygon.interiors]
    for ring in interior_basin_rings:
        ax1.plot(*ring, color='black', linestyle='-', 
                 transform=ccrs.PlateCarree(), linewidth=2)

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.scatter(basin_lon_reg, basin_lat_reg, marker='X', color='red', 
                label='upstream basin', linewidth=2, s=40)
    ax2.scatter(lon_high_flow, lat_high_flow, marker='X', color='green', 
                label='high flow point', linewidth=2, s=80)
    ax2.plot(*basin_polygon.exterior.xy, color='black', 
             label='upstream basin\nboundary', transform=ccrs.PlateCarree(), 
             linewidth=2)
    for ring in interior_basin_rings:
        ax2.plot(*ring, color='black', linestyle='-', 
                 transform=ccrs.PlateCarree(), linewidth=2)
    ax1.scatter(basin_lon_reg, basin_lat_reg, marker='X', color='red', 
                label='upstream basin', linewidth=2, s=300)
    ax1.scatter(lon_high_flow, lat_high_flow, marker='X', color='green', 
                label='high flow point', linewidth=2, s=300)
    ax1.coastlines(linewidth=2)
    ax2.set_extent([-170, -50, 5, 85])
    ax2.coastlines(linewidth=2)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)

    # Add rivers to both plots
    ax1.add_feature(cfeature.RIVERS, linewidth=2, edgecolor='blue', 
                    alpha=0.5)
    ax1.add_feature(cfeature.STATES, linewidth=1, edgecolor='black', 
                    alpha=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black', 
                    alpha=0.5)
    ax2.add_feature(cfeature.STATES, linewidth=1, edgecolor='black', 
                    alpha=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black', 
                    alpha=0.5)
    ax2.add_feature(cfeature.RIVERS, linewidth=2, edgecolor='blue', 
                    alpha=0.5)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def plot_streamflow_timeseries(streamflow_data: xr.DataArray, 
                               precip_data: xr.DataArray,
                               ar_precip_data: xr.DataArray, 
                               melt_data: xr.DataArray,
                               high_flow_mask: xr.DataArray, 
                               high_flow_time: np.datetime64,
                               location: tuple, save_path: str = None):
    """
    Plots the time series of daily streamflow for a given location, 
    marking the identified high flow events.

    Parameters:
        streamflow_data (xr.DataArray): Time series of streamflow with 
                                          dimensions (time).
        precip_data (xr.DataArray): Time series of precipitation with 
                                     dimensions (time).
        ar_precip_data (xr.DataArray): Time series of AR precipitation with 
                                        dimensions (time).
        melt_data (xr.DataArray): Time series of melt with dimensions (time).
        high_flow_mask (xr.DataArray): Boolean mask indicating high flow 
                                        events with the same dimensions as 
                                        streamflow_data.
        location (tuple): A tuple of (latitude, longitude) specifying the 
                          location to plot.
        save_path (str, optional): File path to save the plot. If None, 
                                    the plot is displayed.
    """
    # Extract time series data
    time = streamflow_data['time'].values
    time_window = np.arange(high_flow_time - np.timedelta64(5, 'D'), 
                            high_flow_time + np.timedelta64(10, 'D'), 
                            dtype='datetime64[D]')
    lat, lon = location
    streamflow_values = streamflow_data.sel(time=time_window).values
    precip_values = precip_data.sel(time=time_window).values
    ar_precip_values = ar_precip_data.sel(time=time_window).values
    melt_values = melt_data.sel(time_window).values
    high_flow_mask = high_flow_mask.sel(time=time_window, lat=lat, 
                                         lon=lon).values
    high_flow_events = time_window[high_flow_mask]
    high_flow_value = streamflow_data.sel(time=high_flow_time).values

    # Create the plot
    fig = plt.figure(figsize=(10, 5))
    # Create primary axis for streamflow
    ax1 = plt.gca()
    ax1.plot(time_window, streamflow_values, label='Streamflow', 
             color='black', linewidth=2)
    ax1.set_ylabel('Streamflow [kg day$^{-1}$]', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # Create secondary axis for precipitation, AR precipitation, and melt
    ax2 = ax1.twinx()
    ax2.plot(time_window, precip_values - ar_precip_values, 
             label='non-AR Precipitation', color='mediumblue', linewidth=2)
    ax2.plot(time_window, ar_precip_values, label='AR Precipitation', 
             color='cornflowerblue', linewidth=2)
    ax2.plot(time_window, melt_values, label='Melt', color='purple', 
             linewidth=2)
    ax2.set_ylabel('Precip & Melt [kg day$^{-1$]', color='mediumblue', 
                   fontsize=16)
    ax2.tick_params(axis='y', labelcolor='mediumblue', labelsize=14)
    ax1.scatter(high_flow_time, high_flow_value, color='red', 
                marker='o', label='High Flow Event', s=50)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20)
    ax1.legend(loc='upper left', fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)

    # Increase thickness of axis lines
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)

    plt.tight_layout()
    fig.suptitle(f'Downstream streamflow and upstream sources', 
                  fontsize=20, y=1.)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Streamflow timeseries plot saved to {save_path}", 
              flush=True)
    else:
        plt.show()


def calculate_upstream_variables(daily_data_reg, land_area, 
                                 upstream_basin_mask, lat_high_flow_reg, 
                                 lon_high_flow_reg, precip_var, 
                                 ar_precip_var, snow_var, 
                                 frozen_precip_var, 
                                 monthly_mean_flow_velocity, 
                                 monthly_mean_streamflow):
    upstream_land_area = land_area.where(upstream_basin_mask)
    # Upstream precipitation
    upstream_precip = daily_data_reg[precip_var].where(
        upstream_basin_mask) * 86400
    # Upstream AR precipitation
    upstream_precip_sum = (upstream_precip * upstream_land_area).sum(
        ['lat', 'lon']).load()
    upstream_precip_var = upstream_precip.var(['lat', 'lon']).load()
    upstream_ar_precip = daily_data_reg[ar_precip_var].where(
        upstream_basin_mask) * 86400
    upstream_ar_precip_sum = (upstream_ar_precip * upstream_land_area).sum(
        ['lat', 'lon']).load()
    upstream_ar_precip_var = upstream_ar_precip.var(['lat', 'lon']).load()
    # Upstream frozen precip and snow cover
    upstream_frozen_precip = daily_data_reg[frozen_precip_var].where(
        upstream_basin_mask) * 86400
    upstream_snow_cover = daily_data_reg[snow_var].where(upstream_basin_mask)
    upstream_snow_cover_change = upstream_snow_cover.differentiate(
        'time') * 1e9 * 86400
    # Upstream melt, note that sign flips so melt becomes positive
    upstream_melt = (-(upstream_snow_cover_change - upstream_frozen_precip) * 
                     upstream_land_area)
    upstream_melt_sum = upstream_melt.sum(['lat', 'lon']).load()
    upstream_melt_var = upstream_melt.var(['lat', 'lon']).load()
    # High streamflow
    streamflow = (daily_data_reg.sel(
        lat=lat_high_flow_reg, lon=lon_high_flow_reg)['rv_o_h2o'] *
                  land_area.sel(
                      lat=lat_high_flow_reg, 
                      lon=lon_high_flow_reg)).load() * 86400
    # Upstream mean streamflow
    monthly_mean_upstream_flow_velocity_mean = monthly_mean_flow_velocity.where(
        upstream_basin_mask).mean(['lat', 'lon'])
    monthly_mean_streamflow = (monthly_mean_streamflow.sel(
        lat=lat_high_flow_reg, lon=lon_high_flow_reg) *
                                land_area.sel(
                                    lat=lat_high_flow_reg, 
                                    lon=lon_high_flow_reg)).load() * 86400

    return (upstream_precip_sum, upstream_ar_precip_sum, upstream_melt_sum,
            upstream_precip_var, upstream_ar_precip_var, upstream_melt_var,
            streamflow, monthly_mean_upstream_flow_velocity_mean, 
            monthly_mean_streamflow, upstream_land_area.sum())


def store_correlation_results_to_geodataframe(results, output_filepath):
    """
    Stores the correlation analysis results into a GeoDataFrame and saves 
    it to a file.

    Parameters:
        results (list of dict): List containing dictionaries with correlation 
                                analysis results for each high flow event.
        output_filepath (str): File path to save the GeoDataFrame (e.g., 
                               'results/correlation_results.geojson').

    The GeoDataFrame includes the following columns:
        - high_flow_event (int): Identifier for the high flow event.
        - time (numpy.datetime64): Time of the high flow event.
        - lon_reg (float): Longitude on the regular grid.
        - lat_reg (float): Latitude on the regular grid.
        - lon_cubic (float): Longitude on the cubic grid.
        - lat_cubic (float): Latitude on the cubic grid.
        - upstream_basin_shape (Polygon): Shapely Polygon representing the 
                                            upstream basin.
        - high_flow_streamflow (float): Streamflow on the high flow day.
        - upstream_length_scale (float): Calculated upstream length scale.
        - upstream_flow_velocity_mean (float): Mean upstream flow velocity.
        - scaled_upstream_flow_velocity_mean (float): Scaled upstream flow 
                                                      velocity mean.
        - streamflow_timescale (int): Streamflow timescale in days.
        - best_corr_lag_days (int, optional): Best lag days for correlation.
        - best_corr_var (str, optional): Variable with the best correlation 
                                          ('precip', 'ar_precip', or 'melt').
        - best_corr_value (float, optional): Best correlation coefficient 
                                              value.
        - best_corr_var_window_json (str, optional): JSON string of window 
                                                      values of best correlated 
                                                      variable.
        - streamflow_window_values_json (str, optional): JSON string of 
                                                         streamflow window values.
    """

    # Prepare data for GeoDataFrame
    data = []
    for event in results:
        event_data = {
            'high_flow_event': event['event'],
            'time': event['time'],
            'lon': float(event['lon']),
            'lat': float(event['lat']),
            'upstream_basin_shape': event['upstream_basin_shape'],
            'upstream_basin_area': float(event['upstream_basin_area']),
            'high_flow_streamflow': float(event['high_flow_streamflow']),
            'upstream_length_scale': float(event['upstream_length_scale']),
            'upstream_flow_velocity_mean': float(
                event['upstream_flow_velocity_mean']),
            'scaled_upstream_flow_velocity_mean': float(
                event['scaled_upstream_flow_velocity_mean']),
            'streamflow_timescale': int(event['streamflow_timescale']),
            'best_corr_var': event['correlation']['best_corr_var']
        }

        # Add correlation results if available
        event_data.update({
            key: prepare_data_for_gdf(value) 
            for key, value in event['correlation'].items()
            if isinstance(value, dict)
        })

        data.append(event_data)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry='upstream_basin_shape')

    # Set coordinate reference system (CRS) if known, e.g., WGS84
    gdf.set_crs(epsg=4326, inplace=True)

    # Save to file
    gdf.to_file(output_filepath, driver='GeoJSON')
    print(f"Correlation results saved to {output_filepath}", flush=True)


def prepare_data_for_gdf(data_dict):
    prepared_data_dict = {}
    for key, data in data_dict.items():
        if data is not None and (type(data) == np.float64):
            prepared_data_dict[key] = float(data)
        elif data is not None and (type(data) == np.int64):
            prepared_data_dict[key] = int(data)
        elif data is not None and (type(data) == np.ndarray):
            prepared_data_dict[key] = json.dumps(data.tolist())
        else:
            prepared_data_dict[key] = data
    return prepared_data_dict


def read_high_flow_events(filepath):
    """
    Reads high flow events from a GeoJSON file, including any stored lists.

    Parameters:
        filepath (str): Path to the GeoJSON file containing high flow events

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the high flow events with 
                          parsed data
    """
    # Read the GeoJSON file with lists as strings
    gdf = gpd.read_file(filepath, converters={
        'best_corr_var_window': lambda x: np.array(json.loads(x)) 
            if isinstance(x, str) else None,
        'streamflow_window': lambda x: np.array(json.loads(x)) 
            if isinstance(x, str) else None
    })

    return gdf


def print_report_of_correlation_results(i_event, correlation, n_events):
    print(
        f"Correlation results for high flow event {i_event + 1}/{n_events}:",
        flush=True
    )
    if correlation['best_corr_var'] == 'ar_precip':
        print(
            f"Best correlation with AR precipitation: "
            f"{correlation['best_corr_value']['best_corr_value_ar_precip']:.2f} "
            f"at lag {correlation['best_corr_lag_days']['best_corr_lag_days_ar_precip']} days\n",
            flush=True
        )
    elif correlation['best_corr_var'] == 'precip':
        print(
            f"Best correlation with precipitation: "
            f"{correlation['best_corr_value']['best_corr_value_precip']:.2f} "
            f"at lag {correlation['best_corr_lag_days']['best_corr_lag_days_precip']} days\n",
            flush=True
        )
    elif correlation['best_corr_var'] == 'melt':
        print(
            f"Best correlation with melt: "
            f"{correlation['best_corr_value']['best_corr_value_melt']:.2f} "
            f"at lag {correlation['best_corr_lag_days']['best_corr_lag_days_melt']} days\n",
            flush=True
        )
    elif correlation['best_corr_var'] == 'ar_precip_melt':
        print(
            f"Best correlation with AR precipitation-melt: "
            f"{correlation['best_corr_value']['best_corr_value_ar_precip_melt']:.2f} "
            f"at lag {correlation['best_corr_lag_days']['best_corr_lag_days_ar_precip_melt']} days\n",
            flush=True
        )
    elif correlation['best_corr_var'] == 'precip_melt':
        print(
            f"Best correlation with precipitation-melt: "
            f"{correlation['best_corr_value']['best_corr_value_precip_melt']:.2f} "
            f"at lag {correlation['best_corr_lag_days']['best_corr_lag_days_precip_melt']} days\n",
            flush=True
        )
    else:
        print("No significant correlation found.\n", flush=True)


def process_high_flow_event(args):
    """
    Processes a single high flow event.

    Parameters:
        args (tuple): Tuple containing necessary arguments for processing.

    Returns:
        dict: Result of the correlation analysis for the event.
    """
    (
        i_high_flow, event_coords, daily_data_reg, land_area,
        upstream_basin_mask, precip_var, ar_precip_var, snow_var, 
        frozen_precip_var, monthly_mean_flow_velocity_ds, 
        monthly_mean_streamflow_ds, flow_var, do_test_plots, 
        high_flow_mask, n_events, verbose
    ) = args

    # Get high flow event coordinates
    time_high_flow = np.datetime64(event_coords[0])
    lat_high_flow_reg = event_coords[1]
    lon_high_flow_reg = event_coords[2]
    if verbose:
        print(f"Processing high flow event {i_high_flow + 1}/{n_events} "
              f"at {time_high_flow}", flush=True)
        print(f"Lat: {lat_high_flow_reg}, Lon: {lon_high_flow_reg}", 
              flush=True)
    # Get upstream basin mask
    high_flow_upstream_basin_point = upstream_basin_mask.point.where(
        (upstream_basin_mask.point_lat == lat_high_flow_reg) &
        (upstream_basin_mask.point_lon == lon_high_flow_reg)
    ).dropna('point')
    high_flow_upstream_basin_mask = upstream_basin_mask.sel(
        point=high_flow_upstream_basin_point).upstream_basin_mask.squeeze()
    if high_flow_upstream_basin_mask.sum() == 0:
        print(
            f"No upstream basin found for point {lat_high_flow_reg}, "
            f"{lon_high_flow_reg}\nSkipping event.", flush=True
        )
        return None
    # Get tuples of upstream basin points
    upstream_point_tuples = np.array([
        (lon, lat) for lon in high_flow_upstream_basin_mask.where(
            high_flow_upstream_basin_mask, drop=True).lon.values
        for lat in high_flow_upstream_basin_mask.where(
            high_flow_upstream_basin_mask, drop=True).lat.values
        if high_flow_upstream_basin_mask.sel(lon=lon, lat=lat, 
                                              method='nearest').values
    ])
    print(f"Upstream basin size: {len(upstream_point_tuples)}", 
          flush=True)
    # Create Polygon of upstream basin
    high_flow_upstream_basin_multipoint = MultiPoint(upstream_point_tuples)
    upstream_basin_polygon = high_flow_upstream_basin_multipoint.buffer(
        0.55, cap_style='square', join_style='mitre').buffer(
        -0.25, cap_style='square', join_style='mitre')

    # Select month of high flow event from monthly mean flow velocity 
    # and streamflow
    monthly_mean_flow_velocity = monthly_mean_flow_velocity_ds.sel(
        month=time_high_flow.astype(object).month)
    monthly_mean_streamflow = monthly_mean_streamflow_ds.sel(
        month=time_high_flow.astype(object).month)

    # Extract upstream precipitation, snow cover change and streamflow
    (
        upstream_precip_sum, upstream_ar_precip_sum, upstream_melt_sum,
        upstream_precip_var, upstream_ar_precip_var, upstream_melt_var,
        streamflow, upstream_flow_velocity_mean, 
        monthly_mean_streamflow_val, upstream_land_area_sum
    ) = calculate_upstream_variables(
        daily_data_reg, land_area, high_flow_upstream_basin_mask,
        lat_high_flow_reg, lon_high_flow_reg, precip_var, ar_precip_var,
        snow_var, frozen_precip_var, monthly_mean_flow_velocity, 
        monthly_mean_streamflow
    )

    if do_test_plots:
        plot_upstream_area_for_high_flow_point(
            upstream_point_tuples[:, 0], upstream_point_tuples[:, 1],
            lon_high_flow_reg, lat_high_flow_reg, upstream_basin_polygon,
            save_path='plots/high_low_flow_stat_plots/'
                      f'attribution_tests/upstream_basin_map.png')
        plot_streamflow_timeseries(
            streamflow_data=streamflow, precip_data=upstream_precip_sum,
            ar_precip_data=upstream_ar_precip_sum, melt_data=upstream_melt_sum,
            high_flow_mask=high_flow_mask, high_flow_time=time_high_flow,
            location=(lat_high_flow_reg, lon_high_flow_reg),
            save_path=f'plots/high_low_flow_stat_plots/'
                      f'attribution_tests/annual_streamflow_timeseries.png'
        )

    # Calculate length scale of upstream basin and streamflow timescale
    upstream_length_scale = np.sqrt(upstream_land_area_sum / np.pi)  # m^2
    high_stream_flow = streamflow.sel(time=time_high_flow)
    scaled_upstream_flow_velocity_mean = upstream_flow_velocity_mean * (
            high_stream_flow / monthly_mean_streamflow_val
    )
    try:
        streamflow_timescale = int(np.ceil(
            upstream_length_scale / scaled_upstream_flow_velocity_mean / 86400))  # days
        lag_days = np.arange(streamflow_timescale - streamflow_timescale,
                             streamflow_timescale + streamflow_timescale + 1)
    except:
        print(
            f"Error calculating streamflow timescale for event {i_high_flow + 1}/{n_events}",
            flush=True
        )
        return None
    # Perform lagged correlation
    correlation = perform_lagged_correlation(
        upstream_precip_sum, upstream_ar_precip_sum, upstream_melt_sum,
        upstream_precip_var, upstream_ar_precip_var, upstream_melt_var,
        streamflow, time_high_flow, lag_days=lag_days, 
        temporal_window=streamflow_timescale, plot_lagged_timeseries=do_test_plots
    )
    if correlation['best_corr_var'] == 'ar_precip':
        print('STOP HERE')

    if verbose:
        print_report_of_correlation_results(i_high_flow, correlation, n_events)

    # Collect results
    event_result = {
        'event': i_high_flow + 1,
        'time': time_high_flow,
        'lon': lon_high_flow_reg,
        'lat': lat_high_flow_reg,
        'upstream_basin_shape': upstream_basin_polygon,
        'upstream_basin_area': upstream_land_area_sum,
        'high_flow_streamflow': streamflow.sel(time=time_high_flow).values,
        'upstream_length_scale': upstream_length_scale,
        'upstream_flow_velocity_mean': upstream_flow_velocity_mean.values,
        'scaled_upstream_flow_velocity_mean': scaled_upstream_flow_velocity_mean.values,
        'streamflow_timescale': streamflow_timescale,
        'correlation': correlation
    }

    return event_result


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process high flow events '
                                     'for a given year and experiment')
    parser.add_argument('--year', type=int, help='Year to process', 
                        default=2018)
    parser.add_argument('--exp_name', type=str, help='Experiment name', 
                        default='c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min')
    parser.add_argument('--use_multiprocessing', type=bool, 
                        help='Enable multiprocessing', default=False)
    parser.add_argument('--verbose', type=bool, help='Enable verbose output', 
                        default=True)
    import warnings
    warnings.filterwarnings("ignore")
    # Parse arguments
    args = parser.parse_args()

    # Extract arguments
    year = args.year
    exp_name = args.exp_name
    use_multiprocessing = args.use_multiprocessing
    verbose = args.verbose
    base_path = '/archive/Ming.Zhao/awg/2023.04/'
    gfdl_processor = 'gfdl.ncrc5-intel23-classic-prod-openmp'
    flow_var = 'rv_o_h2o'
    precip_var = 'prli'
    frozen_precip_var = 'prsn'
    ar_precip_var = 'ar_prli'
    snow_var = 'snw'
    variables = [flow_var, precip_var, snow_var, 'pr', frozen_precip_var]
    ar_condition = True
    min_ar_pr_threshold = 1 / 86400
    low_pr_value = 0
    min_pr_var = precip_var
    ar_masked_vars = [precip_var]
    do_test_plots = False

    # Load daily data
    daily_data_reg = lon_180_to_360(sel_na_land(load_model_data(
        base_path, year, variables, exp_name, ar_condition, 
        min_ar_pr_threshold, low_pr_value, min_pr_var, ar_masked_vars, 
        gfdl_processor, lon_180=True,
    ))).load()

    # Load flow threshold
    flow_threshold = xr.open_dataarray(
        f'/archive/Marc.Prange/discharge_statistics/'
        f'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/flow_1p0_year-1_threshold_'
        f'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_1951_2020.nc')
    flow_threshold = lon_180_to_360(sel_na_land(flow_threshold))

    # Load clim mean flow velocities
    monthly_mean_flow_velocity_ds = xr.open_mfdataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/river/av/monthly_42yr/river.1979-2020.*.nc')['rv_veloc'].groupby(
        'time.month').mean('time').load()
    monthly_mean_streamflow_ds = xr.open_mfdataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/river/av/monthly_42yr/river.1979-2020.*.nc')['rv_o_h2o'].groupby(
        'time.month').mean('time').load()
    # Static land area
    land_static = xr.open_dataset(
        f'{base_path}{exp_name}/gfdl.ncrc5-intel23-classic-prod-openmp/'
        f'pp/land/land.static.nc')
    land_area = land_static.land_area

    # Upstream basin lookup
    upstream_basin_mask = xr.open_dataset(
        f'/archive/m2p/high_flow_event_attribution/'
        f'upstream_basin_lookup/upstream_basin_lookup_north_america.nc')
    upstream_basin_mask = upstream_basin_mask.rename({'lat_reg': 'lat', 
                                                      'lon_reg': 'lon'})

    # Calculate high flow mask
    high_flow_mask = calculate_flow_extreme(daily_data_reg, flow_var, 
                                             'high', flow_threshold).load()

    # Get high flow event coordinates
    high_flow_time_lat_lon = get_true_coordinates(high_flow_mask)

    # Prepare arguments for multiprocessing
    args_list = [
        (
            i_high_flow, event_coords, daily_data_reg, land_area,
            upstream_basin_mask, precip_var, ar_precip_var, snow_var, 
            frozen_precip_var, monthly_mean_flow_velocity_ds, 
            monthly_mean_streamflow_ds, flow_var, do_test_plots, 
            high_flow_mask, len(high_flow_time_lat_lon), verbose
        )
        for i_high_flow, event_coords in enumerate(high_flow_time_lat_lon)
    ]

    results = []
    if use_multiprocessing:
        print(f"Starting multiprocessing with {cpu_count()} cores...", 
              flush=True)
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(
                process_high_flow_event, args_list,
                chunksize=len(args_list) // cpu_count()
            ), total=len(args_list), dynamic_ncols=True))
    else:
        print("Starting single-threaded processing...", flush=True)
        for args in tqdm(args_list, desc="Processing high flow events", 
                         dynamic_ncols=True):
            result = process_high_flow_event(args)
            results.append(result)

    # Drop None entries in results
    results = [result for result in results if result is not None]
    store_correlation_results_to_geodataframe(
        results,
        output_filepath=f'/archive/m2p/high_flow_event_attribution/'
                        f'{exp_name}/updated_correlation_results_{year}.geojson'
    )

if __name__ == "__main__":
    main()
