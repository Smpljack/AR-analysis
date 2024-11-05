import geopandas as gpd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from basin_scale_analysis import (
    create_basin_geometries,
    load_static_cubic_data,
    lon360_to_lon180
)

def read_high_flow_data(paths):
    """
    Read high flow event attribution data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        GeoDataFrame: GeoDataFrame containing high flow event data with geometry.
    """
    gdf = pd.concat([gpd.read_file(path) for path in paths])
    return gdf

def assign_basin_id(events_gdf, basin_gdf):
    """
    Assign basin_id to each high flow event based on its location.

    Args:
        events_gdf (GeoDataFrame): GeoDataFrame containing high flow events.
        basin_gdf (GeoDataFrame): GeoDataFrame containing basin geometries.

    Returns:
        GeoDataFrame: Events GeoDataFrame with an additional 'basin_id' column.
    """
    # Spatial join to assign basin_id
    merged_gdf = gpd.sjoin(events_gdf, basin_gdf[['basin_id', 'geometry']], how='left', op='within')
    # Drop unnecessary columns from spatial join
    merged_gdf = merged_gdf.drop(columns=['index_right'])
    return merged_gdf

def process_event_data(data):
    """
    Process high flow event data to calculate total events, attributed events, and their fraction per basin.

    Args:
        data (GeoDataFrame): GeoDataFrame containing high flow event data with basin_id.
        start_year (int, optional): Start year for filtering. Defaults to 1980.
        end_year (int, optional): End year for filtering. Defaults to 2018.

    Returns:
        DataFrame: DataFrame with basin_id, total_events, attributed_events, and fraction.
    """

    # Group by basin_id and calculate total and attributed events
    event_counts = data.groupby('basin_id').size().rename('total_events')
    attributed_mask = [True if var is not None else False for var in data['best_corr_var']]
    precip_mask = [True if var == 'precip' else False for var in data['best_corr_var']]
    ar_precip_mask = [True if var == 'ar_precip' else False for var in data['best_corr_var']]
    melt_mask = [True if var == 'melt' else False for var in data['best_corr_var']]
    data['attributed'] = attributed_mask
    data['precip'] = precip_mask
    data['ar_precip'] = ar_precip_mask
    data['melt'] = melt_mask
    # Combine counts into a single DataFrame
    combined = event_counts
    for var in ['attributed', 'precip', 'ar_precip', 'melt']:
        combined = pd.concat([combined, data.groupby('basin_id')[var].sum().rename(f'{var}_events')], axis=1).fillna(0) 

    combined['event_to_attributed_fraction'] = (combined['attributed_events'] / combined['total_events']) * 100
    combined['precip_to_attributed_fraction'] = (combined['precip_events'] / combined['attributed_events']) * 100
    combined['ar_precip_to_attributed_fraction'] = (combined['ar_precip_events'] / combined['attributed_events']) * 100
    combined['melt_to_attributed_fraction'] = (combined['melt_events'] / combined['attributed_events']) * 100
    combined['ar_precip_to_total_precip_fraction'] = (combined['ar_precip_events'] / (combined['precip_events'] + combined['ar_precip_events'])) * 100
    combined = combined.reset_index()

    return combined

def merge_with_geometry(counts_df, basin_gdf):
    """
    Merge event counts with basin geometries.

    Args:
        counts_df (DataFrame): DataFrame with event counts.
        basin_gdf (GeoDataFrame): GeoDataFrame with basin geometries.

    Returns:
        GeoDataFrame: Merged GeoDataFrame.
    """
    merged_gdf = basin_gdf.merge(counts_df, on='basin_id', how='left')
    merged_gdf[['total_events', 'attributed_events', 'fraction']] = merged_gdf[['total_events', 'attributed_events', 'fraction']].fillna(0)
    return merged_gdf

def plot_event_maps(merged_gdf, region_extent, output_path):
    """
    Plot maps of total events, attributed events, and their fraction per basin.

    Args:
        merged_gdf (GeoDataFrame): GeoDataFrame with merged event counts and geometries.
        region_extent (list): [min_lon, max_lon, min_lat, max_lat] defining the map extent.
        output_path (str): Path to save the output figure.
    """
    # Define projection
    projection = ccrs.PlateCarree()

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': projection})

    # Define plotting parameters for each map
    plot_configs = [
        {
            'column': 'total_events',
            'cmap': 'OrRd',
            'title': 'Total High Flow Events (1980-2018)',
            'legend_label': 'Number of Events',
            'vmin': 0,
            'vmax': merged_gdf['total_events'].max()
        },
        {
            'column': 'attributed_events',
            'cmap': 'BuGn',
            'title': 'Attributed High Flow Events',
            'legend_label': 'Number of Attributed Events',
            'vmin': 0,
            'vmax': merged_gdf['attributed_events'].max()
        },
        {
            'column': 'event_to_attributed_fraction',
            'cmap': 'YlOrBr',
            'title': 'Fraction of Attributed Events (%)',
            'legend_label': 'Fraction (%)',
            'vmin': 0,
            'vmax': 100
        }
    ]

    for ax, config in zip(axes, plot_configs):
        merged_gdf.plot(column=config['column'],
                        cmap=config['cmap'],
                        linewidth=0.8,
                        ax=ax,
                        edgecolor='0.8',
                        legend=False,
                        vmin=config['vmin'],
                        vmax=config['vmax'])

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=config['cmap'], norm=plt.Normalize(vmin=config['vmin'], vmax=config['vmax']))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.04)
        cbar.set_label(config['legend_label'])

        # Add features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.LAKES)
        ax.set_extent(region_extent, crs=projection)

        # Set title
        ax.set_title(config['title'], fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def add_basin_id(high_flow_gdf, static_river_data):
    """
    Adds a 'basin_id' column to the high_flow_gdf by spatially joining with basin_gdf based on event locations.

    Args:
        high_flow_gdf (GeoDataFrame): GeoDataFrame containing high flow events with 'lon_cubic' and 'lat_cubic' columns.
        basin_gdf (GeoDataFrame): GeoDataFrame containing basin geometries with 'basin_id' column.

    Returns:
        GeoDataFrame: Updated high_flow_gdf with an additional 'basin_id' column.
    """
    basin_ids = select_points(
        static_river_data.rv_basin, list(zip(high_flow_gdf.lat_cubic, high_flow_gdf.lon_cubic)))
    high_flow_gdf['basin_id'] = basin_ids
    return high_flow_gdf

def select_points(dataarray: xr.DataArray, points: list, tolerance: float = 1e-5) -> list:
    """
    Efficiently select and return the values from a 2D DataArray at specified (lat, lon) points,
    handling NaN values in the latitude and longitude coordinates.

    Args:
        dataarray (xr.DataArray): The input DataArray with 2D 'lat' and 'lon' coordinates
                                   along 'grid_xt' and 'grid_yt'.
        points (list of tuples): List of (lat, lon) points to select.
        tolerance (float): Maximum allowed distance for matching points.

    Returns:
        list: List of values from the DataArray corresponding to the input points.
              Returns None for points not found within tolerance or if the matching grid point has a NaN value.
    """
    # Extract the latitude and longitude as flattened arrays
    lat = dataarray['geolat_t'].values.flatten()
    lon = dataarray['geolon_t'].values.flatten()
    data_values = dataarray.values.flatten()

    # Identify valid grid points (where both lat and lon are not NaN)
    valid_mask = ~np.isnan(lat) & ~np.isnan(lon)
    valid_lat = lat[valid_mask]
    valid_lon = lon[valid_mask]
    valid_data = data_values[valid_mask]

    # Combine valid lat and lon into a single array of points
    grid_points = np.column_stack((valid_lat, valid_lon))

    # Build a KD-Tree for efficient spatial searches using only valid points
    tree = cKDTree(grid_points)

    # Convert the input points to a NumPy array
    query_points = np.array(points)

    # Query the KD-Tree for nearest neighbors within the specified tolerance
    distances, indices = tree.query(query_points, distance_upper_bound=tolerance)

    # Initialize the list to hold selected values
    selected_values = []

    for dist, idx in zip(distances, indices):
        if dist != np.inf:
            # Retrieve the corresponding value from the valid_data
            value = valid_data[idx]
            selected_values.append(value.item() if isinstance(value, np.generic) else value)
        else:
            # Append None if no match is found within tolerance
            selected_values.append(None)

    return selected_values

def main():
    # File paths
    high_flow_data_paths = ['/archive/Marc.Prange/high_flow_event_attribution/'
                            'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/'
                            f'correlation_results_{year}.geojson' for year in range(1980, 2020)]
    output_figure_path = 'plots/high_flow_event_maps.png'

    # Define CONUS region extent [min_lon, max_lon, min_lat, max_lat]
    conus_extent = [-125, -66.5, 24, 50]

    # Read high flow event data
    high_flow_data = read_high_flow_data(high_flow_data_paths)
    # Convert longitudes from 0-360 to -180-180 range
    high_flow_data['lon_reg'] = np.where(high_flow_data['lon_reg'] > 180, 
                                        high_flow_data['lon_reg'] - 360, 
                                        high_flow_data['lon_reg'])
    high_flow_data['lon_cubic'] = np.where(high_flow_data['lon_cubic'] > 180,
                                          high_flow_data['lon_cubic'] - 360,
                                          high_flow_data['lon_cubic'])

    # Load static river data to get basin geometries
    exp_name = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min'
    tiles = ['tile3', 'tile5']
    static_river_data = load_static_cubic_data(exp_name, sphere='river', tiles=tiles)
    static_land_data = load_static_cubic_data(exp_name, sphere='land', tiles=tiles)
    static_river_data = lon360_to_lon180(static_river_data.assign_coords(geolon_t=static_land_data['geolon_t'],
                                                        geolat_t=static_land_data['geolat_t']).load())
    if static_river_data is None:
        print("No static river data loaded. Exiting.")
        return

    # Assign basin_id to each high flow event based on location
    high_flow_with_basin = add_basin_id(high_flow_data, static_river_data)

    # Process event data
    counts_df = process_event_data(high_flow_with_basin).set_index('basin_id')

    # Create basin geometries
    flattened_basin_data = pd.DataFrame({
        'basin_id': static_river_data.rv_basin.values.flatten(),
        'lon': static_river_data.geolon_t.values.flatten(),
        'lat': static_river_data.geolat_t.values.flatten(),
        'land_area': static_land_data.land_area.values.flatten(),
    })
    basin_geometries = create_basin_geometries(
        counts_df, flattened_basin_data, static_land_data.land_area, static_river_data.rv_basin, 
        min_basin_area=50000)
    # Merge counts with basin geometry
    basin_counts_gdf = gpd.GeoDataFrame(
        counts_df,
        geometry=basin_geometries,
        crs=ccrs.PlateCarree()
    )

    # Plot maps
    plot_event_maps(basin_counts_gdf, conus_extent, output_figure_path)

    print(f"High flow event maps saved to {output_figure_path}")

if __name__ == "__main__":
    main()