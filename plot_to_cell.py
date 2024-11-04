#!/usr/bin/env python
from __future__ import print_function

import numpy      as np
import netCDF4    as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import xarray     as xr

import grdc_util as grdcu
import data_util as du


def find_upstream_ij(rv_dir, idown, jdown):
    rv_dir_to_dij = {
        0:(0,0),
        1:(1,0),
        2:(1,-1),
        3:(0,-1),
        4:(-1,-1),
        5:(-1,0),
        6:(-1,1),
        7:(0,1),
        8:(1,1)
        }
    upstream_ij_list = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            upstream_dir = rv_dir[idown+di, jdown+dj]
            if ~np.isnan(upstream_dir):
                # print(f'di, dj: {di}, {dj}')
                # print(f'upstream cell direction: {np.array(rv_dir_to_dij[int(upstream_dir)])}')
                if (np.array(rv_dir_to_dij[int(upstream_dir)]) == -np.array((di, dj))).all():
                    upstream_ij_list.append((idown+di, jdown+dj))
    if len(upstream_ij_list) == 0:
        return None
    else:
        return np.array(upstream_ij_list)

def find_basin(rv_dir, idown, jdown):
    basin_ij = np.array([[idown, jdown]]) # Basin starts at downstream point
    upstream_ij = [find_upstream_ij(rv_dir, idown, jdown)] # Get first upstream points
    upstream_ij = [
            v for v in upstream_ij if v is not None]
    while len(upstream_ij) > 0: # Concatenate, as long as there are upstream points 
        upstream_ij = np.concatenate(upstream_ij, axis=0)
        basin_ij = np.concatenate([
            basin_ij, upstream_ij], axis=0)
        # print(basin_ij)
        upstream_ij = [
            find_upstream_ij(rv_dir, i, j) for i, j in 
            zip(upstream_ij[:, 0], upstream_ij[:, 1])
            ] 
        upstream_ij = [
            v for v in upstream_ij if v is not None]
        # if basin_ij.shape[0] > 1000:
        #     print(f'Basin size: {basin_ij.shape[0]}.')
        #     return None
    return basin_ij

        

def _main():
    parser = argparse.ArgumentParser(description='plot river routing')
    parser.add_argument('-v','--verbose', dest='verb', default=0,
        help='increase verbosity', action='count')
    parser.add_argument(
        '-s','--save', nargs=1, metavar='FILENAME',
        help='save figure instead of plotting it on screen.')
    # parser.add_argument(
    #     'input', metavar='FILENAME',
    #     help='input river hydrography file')
    model_run_path = \
        '/archive/Ming.Zhao/awg/2022.03/c192L33_am4p0_amip_HIRESMIP_nudge_wind_1951_2020/'
    river_cubic_static_path = [model_run_path + \
        f'gfdl.ncrc4-intel-prod-openmp/pp/river_cubic/river_cubic.static.tile{tile}.nc'
        for tile in [3, 5]]
    land_cubic_static_path = [model_run_path + \
        f'gfdl.ncrc4-intel-prod-openmp/pp/land_cubic/land_cubic.static.tile{tile}.nc'
        for tile in [3, 5]]
    args=parser.parse_args()
    direction={
        0:(0,0),
        1:(0,-1),
        2:(-1,-1),
        3:(-1,0),
        4:(-1,1),
        5:(0,1),
        6:(1,1),
        7:(1,0),
        8:(1,-1)
    }
    dir_t3 = {
        0: (0, 0),
        1: (-1, 0),
        2: (-1, 1),
        3: (0, 1),
        4: (1, 1),
        5: (1, 0),
        6: (1, -1),
        7: (0, -1),
        8: (-1, -1)
    }
    dir_t3_to_t5 = {
        0: 0,
        1: 3, 
        2: 4,
        3: 5,
        4: 6,
        5: 7, 
        6: 8,
        7: 1,
        8: 2
    }
    if args.verb > 0:
        print('using pakages:')
        for package in np,nc,mpl:
            print('    {:>20} : {}'.format(package.__name__,package.__version__))
    rv_data_t3 = xr.open_dataset(river_cubic_static_path[0])
    rv_data_t5 = xr.open_dataset(river_cubic_static_path[1])
    land_data_t3 = xr.open_dataset(land_cubic_static_path[0])
    land_data_t5 = xr.open_dataset(land_cubic_static_path[1])
    # rv_dir_t3 = np.array(
    #     [[dir_t3_to_t5[rv_data_t3.rv_dir.values[j, i]] 
    #       if ~np.isnan(rv_data_t3.rv_dir.values[j, i]) else np.nan 
    #       for i in range(rv_data_t3.rv_dir.shape[1])] 
    #      for j in range(rv_data_t3.rv_dir.shape[0])])
    # rv_data_t3['rv_dir'] = (('grid_yt', 'grid_xt'), rv_dir_t3)
    # rv_data_t3 = rv_data_t3.rename({'grid_xt': 'grid_yt', 'grid_yt': 'grid_xt'})
    rv_data = xr.concat([rv_data_t3, rv_data_t5], dim='grid_yt')
    land_data = xr.concat([land_data_t3, land_data_t5], dim='grid_yt')
    land_data = land_data.where(
        (land_data.geolat_t > 20) & (land_data.geolat_t < 70) & 
        (land_data.geolon_t > -140+360) & (land_data.geolon_t < -60+360))
    rv_data = rv_data.where(
        (land_data.geolat_t > 20) & (land_data.geolat_t < 70) & 
        (land_data.geolon_t > -140+360) & (land_data.geolon_t < -60+360))
    rv_dir = rv_data.rv_dir.T.values
    lat_grid, lon_grid = (land_data.geolat_t.T.values, land_data.geolon_t.T.values)
    lon_grid = np.where(lon_grid > 180, lon_grid-360, lon_grid)
    lat_gauge = 32.5
    lon_gauge = -115
    dist_gauge_grid = grdcu.haversine(lat_grid, lon_grid, lat_gauge, lon_gauge)
    min_dist = np.nanmin(grdcu.haversine(lat_grid, lon_grid, lat_gauge, lon_gauge))
    igrid_gauge, jgrid_gauge = tuple([v[0] for v in np.where(dist_gauge_grid == min_dist)])
    lat_grid_gauge = lat_grid[igrid_gauge, jgrid_gauge]
    lon_grid_gauge = lon_grid[igrid_gauge, jgrid_gauge]
    # print(tocell.shape)
    basin_ij = find_basin(rv_dir=rv_dir, idown=igrid_gauge, jdown=jgrid_gauge)
    # dlon_dlat = np.array([np.gradient(lon_grid.values)[0], np.gradient(lat_grid.values)[0]]).transpose(1, 2, 0)
    di_dj = np.array(
        [[direction[rv_dir[i, j]] if ~np.isnan(rv_dir[i, j]) else (np.nan, np.nan) 
          for j in range(rv_dir.shape[1])]
         for i in range(rv_dir.shape[0])])
    lon_grid[np.isnan(di_dj[:, :, 0])] = np.nan
    lat_grid[np.isnan(di_dj[:, :, 0])] = np.nan
    di_dj_basin = np.array(
        [direction[rv_dir[i, j]] for i, j in zip(basin_ij[:, 0], basin_ij[:, 1])])
    lon_lat_basin = np.array(
        [[lon_grid[i, j], lat_grid[i, j]] for i, j in zip(basin_ij[:, 0], basin_ij[:, 1])])

    fig, ax = plt.subplots(facecolor='w',figsize=(12,12))
    [[ax.arrow(lon_grid[i, j], lat_grid[i, j], di_dj[i, j, 0]/3, di_dj[i, j, 1]/3, 
              head_width=0.1,head_length=0.1,length_includes_head=True) 
      for i in range(di_dj.shape[0])]
    for j in range(di_dj.shape[1])]
    [ax.arrow(lon_lat_basin[i, 0], lon_lat_basin[i, 1], di_dj_basin[i, 0]/3, di_dj_basin[i, 1]/3, 
              color='red', head_width=0.1,head_length=0.1,length_includes_head=True)
    for i in range(di_dj_basin.shape[0])]
    plt.scatter(lon_lat_basin[0, 0], lon_lat_basin[0, 1], marker='o', color='green', s=6)
    ax.set_title('tocell')
    # ax.set_xlabel('x label')
    ax.grid()
    # ax.legend(loc='best')

    # plt.arrow(0.5,0.5,1.0,1.0)

    
    fig.savefig('plots/basin_recursive_lm4_test_colorado_river.png', transparent=False, bbox_inches='tight', dpi=300)
    
if __name__ == '__main__':
    _main()
