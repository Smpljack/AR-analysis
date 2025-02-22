import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from xarray.groupers import BinGrouper
import xarray as xr
from data_util import get_global_mean_dT, lon_360_to_180


def read_geo_file(path):
    return pd.read_json(path)

def load_high_flow_data(
        exp_name, start_year, end_year, specific_high_flow_mask=None):
    high_flow_data_paths = [f'/archive/Marc.Prange/high_flow_event_attribution/'
                           f'{exp_name}/'
                           f'correlation_results_{year}_incl_antecedent_huc_homog_driver_windows'
                           + (f'_{specific_high_flow_mask}' 
                              if specific_high_flow_mask is not None else '') + 
                           f'.json'
                           for year in range(start_year, end_year + 1)]

    with ThreadPoolExecutor() as executor:
        high_flow_data_list = list(tqdm(
            executor.map(read_geo_file, high_flow_data_paths), 
            total=len(high_flow_data_paths)))
    
    # Read high flow event data
    high_flow_data = pd.concat(high_flow_data_list)#read_high_flow_data(high_flow_data_paths)
    # Convert longitudes from 0-360 to -180-180 range
    high_flow_data['lon'] = np.where(high_flow_data['lon'] > 180, 
                                        high_flow_data['lon'] - 360, 
                                        high_flow_data['lon'])
    # Convert huc2_id/huc4_id from int to string with leading zeros
    high_flow_data['huc2_id'] = [f"{val:0{2}}" 
                                 for val in high_flow_data['huc2_id'].values]
    high_flow_data['huc4_id'] = [f"{val:0{4}}" 
                                 for val in high_flow_data['huc4_id'].values]
    # Transform lists 
    for var in ['soilm_sat_1cm_window', 'soilm_sat_4cm_window', 
                'soilm_sat_8cm_window', 'streamflow_window']:
        high_flow_data[var] = high_flow_data[var].apply(string_list_to_list)
    high_flow_data['best_corr_window_var'] = \
        high_flow_data['best_corr_window_var'].apply(
            string_list_to_list_for_dict)
    high_flow_data['zero_shift_window_var'] = \
        high_flow_data['zero_shift_window_var'].apply(
            string_list_to_list_for_dict)
    
    for var in ['best_corr_window_ar_precip', 'best_corr_window_precip', 
                'best_corr_window_melt']:
        high_flow_data[var] = high_flow_data['best_corr_window_var'].apply(
            select_key_from_dict, args=(var,))
    
    for var in ['zero_shift_window_values_precip', 'zero_shift_window_values_ar_precip', 
                'zero_shift_window_values_melt']:
        high_flow_data[var] = high_flow_data['zero_shift_window_var'].apply(
            select_key_from_dict, args=(var,)) 
        
    # Get mean antecedent soil saturation
    for var in ['soilm_sat_1cm_window', 'soilm_sat_4cm_window', 
                'soilm_sat_8cm_window']:
        high_flow_data[var.strip('_window') + '_antec_min'] = \
            high_flow_data[var].apply(get_antecedent_metric, metric='min', 
                                      shift=0)
    high_flow_data['streamflow_window_sum'] = \
        high_flow_data['streamflow_window'].apply(np.sum)
    for var in ['ar_precip', 'precip', 'melt']:
        high_flow_data[f'zero_shift_window_values_{var}_antec_sum'] = \
            pd.Series([get_antecedent_metric(best_corr_window_values, 'sum', 0)
             for best_corr_window_values
             in high_flow_data[f'zero_shift_window_values_{var}']], 
             index=high_flow_data.index)
        high_flow_data[f'zero_shift_window_values_{var}_antec_mean'] = \
            pd.Series([get_antecedent_metric(best_corr_window_values, 'mean', 0)
             for best_corr_window_values
             in high_flow_data[f'zero_shift_window_values_{var}']], 
             index=high_flow_data.index)
        high_flow_data[f'zero_shift_window_values_{var}_antec_max'] = \
            pd.Series([get_antecedent_metric(best_corr_window_values, 'max', 0)
             for best_corr_window_values
             in high_flow_data[f'zero_shift_window_values_{var}']], 
             index=high_flow_data.index)
        high_flow_data[f'zero_shift_window_values_{var}_sum'] = \
            pd.Series([get_window_metric(best_corr_window_values, 'sum')
             for best_corr_window_values
             in high_flow_data[f'zero_shift_window_values_{var}']], 
             index=high_flow_data.index)
        high_flow_data[f'runoff_ratio_window_{var}'] = \
            high_flow_data['streamflow_window_sum']/high_flow_data[f'zero_shift_window_values_{var}_sum']
    return high_flow_data

def match_time_lat_lon_order(ref_data, dist_data):
    ref_mask = np.where(
        pd.merge(
            ref_data, dist_data, on=['time', 'lat', 'lon'], 
            how='left', indicator='Exist').Exist == 'both', 
        True, False)
    dist_mask = np.where(
        pd.merge(
            dist_data, ref_data, on=['time', 'lat', 'lon'], 
            how='left', indicator='Exist').Exist == 'both', 
        True, False)
    sorted_ref_data = ref_data[ref_mask].sort_values(
        ['time', 'lat', 'lon'], kind='heapsort')
    sorted_dist_data = dist_data[dist_mask].sort_values(
        ['time', 'lat', 'lon'], kind='heapsort')
    return sorted_ref_data.reset_index(drop=True), sorted_dist_data.reset_index(drop=True)

def string_list_to_list(string_list):
    if string_list is not None:
        return [float(item.strip()) for item in string_list[1:-1].split(',')]
    else:
        return None

def string_list_to_list_for_dict(string_list_dict):
    return {key: string_list_to_list(val) 
            for key, val in string_list_dict.items()}

def get_antecedent_metric(high_flow_data, metric='mean', shift=0):
    if (high_flow_data is None) or (np.isnan(shift)):
        return None
    antecedent_window_data = high_flow_data[
        int(shift):int(np.ceil(len(high_flow_data)/2)+shift)]
    if metric == 'mean':
        return np.mean(antecedent_window_data)
    elif metric == 'median':
        return np.median(antecedent_window_data)
    elif metric == 'max':
        return np.max(antecedent_window_data) if len(antecedent_window_data) > 0 else np.nan
    elif metric == 'min':
        return np.min(antecedent_window_data) if len(antecedent_window_data) > 0 else np.nan
    elif metric == 'sum':
        return np.sum(antecedent_window_data)
    else:
        raise ValueError(f'Invalid metric: {metric}')

def get_window_metric(high_flow_data, metric='mean', shift=0):
    if (high_flow_data is None) or (np.isnan(shift)):
        return None
    if metric == 'mean':
        return np.mean(high_flow_data)
    elif metric == 'median':
        return np.median(high_flow_data)
    elif metric == 'max':
        return np.max(high_flow_data)
    elif metric == 'min':
        return np.min(high_flow_data)
    elif metric == 'sum':
        return np.sum(high_flow_data)
    else:
        raise ValueError(f'Invalid metric: {metric}')

def get_antecedent_metric_for_dict(high_flow_data_dict, metric='mean'):
    return {key: get_antecedent_metric(val, metric) if val is not None else None
            for key, val in high_flow_data_dict.items()}

def select_key_from_dict(high_flow_data_dict, key):
    return high_flow_data_dict[key]

def scatter_driver_ctrl_vs_driver_p2K(
        ref_data, dist_data):
    ar_precip_event_mask = ref_data['best_corr_var'] == 'ar_precip'
    precip_event_mask = ref_data['best_corr_var'] == 'precip'
    melt_event_mask = ref_data['best_corr_var'] == 'melt'
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (event_type, event_mask) in enumerate(zip(['ar_precip', 'precip', 'melt'], 
                                    [ar_precip_event_mask, precip_event_mask, 
                                     melt_event_mask])):
        driver_ctrl = ref_data[event_mask]['best_corr_window_var_antec'].apply(
            select_key_from_dict, args=(f'best_corr_window_{event_type}',)
        )
        driver_dist = dist_data[event_mask]['best_corr_window_var_antec'].apply(
            select_key_from_dict, args=(f'best_corr_window_{event_type}',)
        )
        axs[i].scatter(driver_ctrl, driver_dist, alpha=0.5)
        axs[i].set_xlabel(f'CTRL')
        axs[i].set_ylabel(f'P2K')
        axs[i].set_title(f'{event_type} antecedent driver')
        xlims = axs[i].get_xlim()
        ylims = axs[i].get_ylim()
        extreme_limits = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
        axs[i].set_ylim(extreme_limits)
        axs[i].set_xlim(extreme_limits)
    plt.tight_layout()
    plt.savefig(
        f'plots/antecedent_conditions/{event_type}_antecedent_driver_scatter.png',
        dpi=300)

def scatter_driver_change_vs_streamflow_change_vs_antecedent_soilm_sat_change_by_region(
        ref_data, dist_data, dT, soilm_depth='8cm'):
    ref_data_west_coast = ref_data[ref_data['lon'] <= -120]
    dist_data_west_coast = dist_data[dist_data['lon'] <= -120]
    ref_data_mountainous_west = ref_data[(ref_data['lon'] > -120) & (ref_data['lon'] <= -100)]
    dist_data_mountainous_west = dist_data[(dist_data['lon'] > -120) & (dist_data['lon'] < -100)]
    ref_data_east = ref_data[(ref_data['lon'] > -100) & (ref_data['lon'] <= -67)]
    dist_data_east = dist_data[(dist_data['lon'] > -100) & (dist_data['lon'] <= -67)]
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    for iregion, (region, ref_data, dist_data) in enumerate(
        [('West Coast', ref_data_west_coast, dist_data_west_coast), 
         ('Mountainous West', ref_data_mountainous_west, dist_data_mountainous_west), 
         ('East', ref_data_east, dist_data_east)]):
        ar_precip_event_mask = ref_data['best_corr_var'] == 'ar_precip'
        precip_event_mask = ref_data['best_corr_var'] == 'precip'
        melt_event_mask = ref_data['best_corr_var'] == 'melt'
        for idriver, (event_type, event_mask) in enumerate(
            zip(['ar_precip', 'precip', 'melt'], 
                [ar_precip_event_mask, precip_event_mask, melt_event_mask])):
            driver_ref = ref_data[event_mask]['best_corr_window_var_antec'].apply(
                select_key_from_dict, args=(f'best_corr_window_{event_type}',)
            )
            driver_dist = dist_data[event_mask]['best_corr_window_var_antec'].apply(
                select_key_from_dict, args=(f'best_corr_window_{event_type}',)
            )
            driver_change = (driver_dist - driver_ref)/driver_ref/dT*100
            streamflow_ref = ref_data[event_mask]['high_flow_streamflow']
            streamflow_dist = dist_data[event_mask]['high_flow_streamflow']
            streamflow_change = (streamflow_dist - streamflow_ref)/streamflow_ref/dT*100
            soilm_sat_ref = ref_data[event_mask][f'soilm_sat_{soilm_depth}_antec']
            soilm_sat_dist = dist_data[event_mask][f'soilm_sat_{soilm_depth}_antec']
            soilm_sat_change = (soilm_sat_dist - soilm_sat_ref)/soilm_sat_ref/dT*100
            
            s = axs[iregion, idriver].scatter(
                driver_change, streamflow_change, c=soilm_sat_change, cmap='RdBu', 
                alpha=0.5, vmin=-20, vmax=20)
            cb = plt.colorbar(s, ax=axs[iregion, idriver])
            cb.set_label(f'Soil sat. change [% K$^{-1}$]')
            axs[iregion, idriver].set_xlabel(f'Driver change [% K$^{-1}$]')
            axs[iregion, idriver].set_ylabel(f'Streamflow change [% K$^{-1}$]')
            axs[iregion, idriver].set_title(f'{event_type} {region}')
            xlims = [-50, 150]
            ylims = [-50, 150]
            extreme_limits = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
            axs[iregion, idriver].set_ylim(extreme_limits)
            axs[iregion, idriver].set_xlim(extreme_limits)
    plt.tight_layout()
    plt.savefig(
        f'plots/antecedent_conditions/driver_streamflow_soilm_sat_{soilm_depth}_change_scatter_per_region.png',
        dpi=300)

def df_to_da(
        df, 
        variables=['time', 'lat', 'lon', 'zero_shift_window_values_precip_antec_sum', 
                   'zero_shift_window_values_ar_precip_antec_sum', 
                   'zero_shift_window_values_melt_antec_sum', 
                   'zero_shift_window_values_precip_antec_mean', 
                   'zero_shift_window_values_ar_precip_antec_mean', 
                   'zero_shift_window_values_melt_antec_mean', 
                   'zero_shift_window_values_precip_antec_max', 
                   'zero_shift_window_values_ar_precip_antec_max', 
                   'zero_shift_window_values_melt_antec_max', 
                   'high_flow_streamflow', 
                   'soilm_sat_1cm_antec_min', 'soilm_sat_4cm_antec_min', 
                   'soilm_sat_8cm_antec_min', 'runoff_ratio_window_precip',
                   'runoff_ratio_window_ar_precip', 'runoff_ratio_window_melt',
                #    'is_high_flow_precip', 'is_high_flow_ar_precip', 'is_high_flow_melt', 
                   'is_high_flow', 'streamflow_timescale', 'best_corr_var',
                   'upstream_basin_area']):
    df = df.reset_index()
    da = df.to_xarray()
    da = da[variables]
    # da = da.assign_coords(index=np.arange(len(da.index)))
    return da

def heatmap_driver_change_vs_streamflow_change_vs_antecedent_soilm_sat_change_by_region(
        ref_data, dist_data, dT, soilm_depth='8cm', 
        bins={'ar_precip': np.linspace(-10, 22, 6),
              'precip': np.linspace(-10, 22, 6),
              'melt': np.linspace(-35, 5, 6)}):
    bin_centers = {key: (bins[key][:-1] + bins[key][1:])/2
                   for key in bins}
    ref_data_west_coast = ref_data[ref_data['lon'] <= -120]
    dist_data_west_coast = dist_data[dist_data['lon'] <= -120]
    ref_data_mountainous_west = ref_data[(ref_data['lon'] > -120) & (ref_data['lon'] <= -100)]
    dist_data_mountainous_west = dist_data[(dist_data['lon'] > -120) & (dist_data['lon'] <= -100)]
    ref_data_east = ref_data[(ref_data['lon'] > -100) & (ref_data['lon'] <= -67)]
    dist_data_east = dist_data[(dist_data['lon'] > -100) & (dist_data['lon'] <= -67)]
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    for iregion, (region, ref_data_region, dist_data_region) in enumerate(
        [('West Coast', ref_data_west_coast, dist_data_west_coast), 
         ('Mountainous West', ref_data_mountainous_west, dist_data_mountainous_west), 
         ('East', ref_data_east, dist_data_east)]):
        ar_precip_event_mask = (ref_data_region['best_corr_var'] == 'ar_precip')
        precip_event_mask = (ref_data_region['best_corr_var'] == 'precip')
        melt_event_mask = (ref_data_region['best_corr_var'] == 'melt')
        for idriver, (event_type, event_mask) in enumerate(
            zip(['ar_precip', 'precip', 'melt'], 
                [ar_precip_event_mask, precip_event_mask, melt_event_mask])):
            ref_da = df_to_da(ref_data_region[event_mask])
            dist_da = df_to_da(dist_data_region[event_mask])
            change_da = (dist_da - ref_da)/ref_da/dT*100
            change_da[f'soilm_sat_{soilm_depth}_antec_min'] = \
                (dist_da - ref_da)[f'soilm_sat_{soilm_depth}_antec_min']/dT*100
            change_da_binned = change_da.groupby(
                {'high_flow_streamflow': BinGrouper(
                    bins=bins[event_type]), 
                 f'best_corr_window_{event_type}_antec_sum': BinGrouper(
                     bins=bins[event_type])}
                 )
            change_da_binned_mean = change_da_binned.mean()
            change_da_binned_count = change_da_binned.count()
            driver_change = change_da_binned_mean[f'best_corr_window_{event_type}_antec_sum']
            streamflow_change = change_da_binned_mean['high_flow_streamflow']
            soilm_sat_change = change_da_binned_mean[f'soilm_sat_{soilm_depth}_antec_min']
            count = change_da_binned_count[f'soilm_sat_{soilm_depth}_antec_min']
            s = axs[iregion, idriver].pcolormesh(
                bin_centers[event_type], bin_centers[event_type], soilm_sat_change.T, 
                cmap='RdBu', alpha=1, vmin=-10, vmax=10)
            axs[iregion, idriver].contour(
                bin_centers[event_type], bin_centers[event_type], count.T, 
                levels=np.linspace(0, count.max(), 10), 
                colors='black', linewidths=0.5)
            cb = plt.colorbar(s, ax=axs[iregion, idriver])
            cb.set_label(f'Soil sat. change [% K$^{-1}$]')
            axs[iregion, idriver].set_xlabel(f'Driver change [% K$^{-1}$]')
            axs[iregion, idriver].set_ylabel(f'Streamflow change [% K$^{-1}$]')
            axs[iregion, idriver].set_title(f'{event_type} {region}')
            xlims = [bins[event_type][0], bins[event_type][-1]]
            ylims = [bins[event_type][0], bins[event_type][-1]]
            extreme_limits = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
            axs[iregion, idriver].set_ylim(extreme_limits)
            axs[iregion, idriver].set_xlim(extreme_limits)
    plt.tight_layout()
    plt.savefig(
        f'plots/antecedent_conditions/driver_streamflow_soilm_sat_{soilm_depth}_change_heatmap_per_region.png',
        dpi=300)

def boxplot_asm_vs_runoff_ratio(
        ref_data, dist_data, soilm_depth='8cm', 
        bins=np.linspace(0, 1, 6)):
    bin_centers = (bins[:-1] + bins[1:])/2
    ref_data_west_coast = ref_data[ref_data['lon'] <= -120]
    dist_data_west_coast = dist_data[dist_data['lon'] <= -120]
    ref_data_mountainous_west = ref_data[(ref_data['lon'] > -120) & (ref_data['lon'] <= -100)]
    dist_data_mountainous_west = dist_data[(dist_data['lon'] > -120) & (dist_data['lon'] <= -100)]
    ref_data_east = ref_data[(ref_data['lon'] > -100) & (ref_data['lon'] <= -67)]
    dist_data_east = dist_data[(dist_data['lon'] > -100) & (dist_data['lon'] <= -67)]
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    for iregion, (region, ref_data_region, dist_data_region) in enumerate(
        [('West Coast', ref_data_west_coast, dist_data_west_coast), 
         ('Mountainous West', ref_data_mountainous_west, dist_data_mountainous_west), 
         ('East', ref_data_east, dist_data_east)]):
        ar_precip_event_mask_ref = (ref_data_region['best_corr_var'] == 'ar_precip')
        precip_event_mask_ref = (ref_data_region['best_corr_var'] == 'precip')
        melt_event_mask_ref = (ref_data_region['best_corr_var'] == 'melt')
        ar_precip_event_mask_dist = (dist_data_region['best_corr_var'] == 'ar_precip')
        precip_event_mask_dist = (dist_data_region['best_corr_var'] == 'precip')
        melt_event_mask_dist = (dist_data_region['best_corr_var'] == 'melt')
        ar_precip_event_mask = ar_precip_event_mask_ref | ar_precip_event_mask_dist
        precip_event_mask = precip_event_mask_ref | precip_event_mask_dist
        melt_event_mask = melt_event_mask_ref | melt_event_mask_dist
        for idriver, (event_type, event_mask) in enumerate(
            zip(['ar_precip', 'precip', 'melt'], 
                [ar_precip_event_mask, precip_event_mask, melt_event_mask])):
            ref_da = df_to_da(ref_data_region[event_mask])
            dist_da = df_to_da(dist_data_region[event_mask])
            ref_da_binned = ref_da.groupby_bins(
                f'soilm_sat_{soilm_depth}_antec_min', 
                bins=bins)
            ref_da_binned_median = ref_da_binned.quantile(0.5)[f'runoff_ratio_window_{event_type}']
            ref_da_binned_qmin = ref_da_binned.quantile(0.05)[f'runoff_ratio_window_{event_type}']
            ref_da_binned_q1 = ref_da_binned.quantile(0.25)[f'runoff_ratio_window_{event_type}']
            ref_da_binned_q3 = ref_da_binned.quantile(0.75)[f'runoff_ratio_window_{event_type}']
            ref_da_binned_qmax = ref_da_binned.quantile(0.95)[f'runoff_ratio_window_{event_type}']
            ref_da_binned_count = ref_da_binned.sum()[f'is_high_flow_{event_type}']
            dist_da_binned = dist_da.groupby_bins(
                f'soilm_sat_{soilm_depth}_antec_min', 
                bins=bins)
            dist_da_binned_median = dist_da_binned.quantile(0.5)[f'runoff_ratio_window_{event_type}']
            dist_da_binned_qmin = dist_da_binned.quantile(0.05)[f'runoff_ratio_window_{event_type}']
            dist_da_binned_q1 = dist_da_binned.quantile(0.25)[f'runoff_ratio_window_{event_type}']
            dist_da_binned_q3 = dist_da_binned.quantile(0.75)[f'runoff_ratio_window_{event_type}']
            dist_da_binned_qmax = dist_da_binned.quantile(0.95)[f'runoff_ratio_window_{event_type}']
            dist_da_binned_count = dist_da_binned.sum()[f'is_high_flow_{event_type}']

            ref_bxpstats = [{
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [median_ref, q1_ref, q3_ref, qmin_ref, qmax_ref])}
                for median_ref, q1_ref, q3_ref, qmin_ref, qmax_ref in zip(
                    ref_da_binned_median, ref_da_binned_q1, ref_da_binned_q3, 
                    ref_da_binned_qmin, ref_da_binned_qmax)]
            dist_bxpstats = [{
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [median_dist, q1_dist, q3_dist, qmin_dist, qmax_dist])}
                for median_dist, q1_dist, q3_dist, qmin_dist, qmax_dist in zip(
                    dist_da_binned_median, dist_da_binned_q1, dist_da_binned_q3, 
                    dist_da_binned_qmin, dist_da_binned_qmax)]
            s = axs[idriver, iregion].bxp(
                bxpstats=ref_bxpstats,
                positions=np.round(bin_centers-0.026, 3),
                boxprops={'color': 'black'},
                widths=0.05,
                showmeans=False, showfliers=False, shownotches=False)
            s = axs[idriver, iregion].bxp(
                bxpstats=dist_bxpstats,
                positions=np.round(bin_centers+0.026, 3),
                boxprops={'color': 'red'},
                widths=0.05,
                showmeans=False, showfliers=False, shownotches=False)
            ax2 = axs[idriver, iregion].twinx()  # Create a new y-axis for the bar plots
            ax2.bar(np.round(bin_centers-0.026, 3), ref_da_binned_count, 
                    width=0.05, color='black', alpha=0.5, label='Ref Count')
            ax2.bar(np.round(bin_centers+0.026, 3), dist_da_binned_count, 
                    width=0.05, color='red', alpha=0.5, label='P2K Count')
            ax2.set_ylabel('Count')
            ax2.legend(loc='upper right')
            ax2.set_ylim([0, 2.2*max(ref_da_binned_count.max(), dist_da_binned_count.max())])
            axs[idriver, iregion].set_xlabel(f'ASM saturation[-]')
            axs[idriver, iregion].set_ylabel(f'Runoff Ratio [-]')
            axs[idriver, iregion].set_title(f'{event_type} {region}')
            axs[idriver, iregion].set_ylim([-1, 1])
            axs[idriver, iregion].set_xlim([-0.1, 1.1])
            axs[idriver, iregion].hlines(0, -0.1, 1.1, color='black', linewidth=0.5)
            axs[idriver, iregion].set_xticks(np.round(bin_centers, 2))
    plt.tight_layout()
    plt.savefig(
        f'plots/antecedent_conditions/boxplot_asm_{soilm_depth}_vs_runoff_ratio.png',
        dpi=300)

def boxplot_high_flow_change_vs_asm_change_by_driver(
        ref_data, dist_data, dT, soilm_depth='8cm'):
    ref_data_west_coast = ref_data[ref_data['lon'] <= -122]
    dist_data_west_coast = dist_data[dist_data['lon'] <= -122]
    ref_data_mountainous_west = ref_data[(ref_data['lon'] > -122) & (ref_data['lon'] <= -100)]
    dist_data_mountainous_west = dist_data[(dist_data['lon'] > -122) & (dist_data['lon'] <= -100)]
    ref_data_east = ref_data[(ref_data['lon'] > -100) & (ref_data['lon'] <= -67)]
    dist_data_east = dist_data[(dist_data['lon'] > -100) & (dist_data['lon'] <= -67)]
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    for iregion, (region, ref_data_region, dist_data_region) in enumerate(
        [('West Coast', ref_data_west_coast, dist_data_west_coast), 
         ('Mountainous West', ref_data_mountainous_west, dist_data_mountainous_west), 
         ('East', ref_data_east, dist_data_east)]):
        ar_precip_event_mask_ref = (ref_data_region['best_corr_var'] == 'ar_precip').values
        precip_event_mask_ref = (ref_data_region['best_corr_var'] == 'precip').values
        melt_event_mask_ref = (ref_data_region['best_corr_var'] == 'melt').values
        ar_precip_event_mask_dist = (dist_data_region['best_corr_var'] == 'ar_precip').values
        precip_event_mask_dist = (dist_data_region['best_corr_var'] == 'precip').values
        melt_event_mask_dist = (dist_data_region['best_corr_var'] == 'melt').values
        
        for idriver, (event_type, event_mask_ref, event_mask_dist) in enumerate(
            zip(['ar_precip', 'precip', 'melt'], 
                [ar_precip_event_mask_ref, precip_event_mask_ref, melt_event_mask_ref],
                [ar_precip_event_mask_dist, precip_event_mask_dist, melt_event_mask_dist])):
            ref_da_only_ref_high_flow = df_to_da(
                ref_data_region[event_mask_ref&~event_mask_dist])
            dist_da_only_ref_high_flow = df_to_da(
                dist_data_region[event_mask_ref&~event_mask_dist])
            ref_da_only_dist_high_flow = df_to_da(
                ref_data_region[~event_mask_ref&event_mask_dist])
            dist_da_only_dist_high_flow = df_to_da(
                dist_data_region[~event_mask_ref&event_mask_dist])
            ref_da_ref_dist_high_flow = df_to_da(
                ref_data_region[event_mask_ref&event_mask_dist])
            dist_da_ref_dist_high_flow = df_to_da(
                dist_data_region[event_mask_ref&event_mask_dist])
            ref_asm_bxpstats = []
            dist_asm_bxpstats = []
            change_driver_bxpstats = []
            high_flow_count = []
            for ref_da, dist_da, high_flow_occ_str in zip(
                [ref_da_only_ref_high_flow, ref_da_only_dist_high_flow, ref_da_ref_dist_high_flow],
                [dist_da_only_ref_high_flow, dist_da_only_dist_high_flow, dist_da_ref_dist_high_flow],
                ["Only CTRL", "Only P2K", "Both"]):
                change_da_driver = (dist_da[f'best_corr_window_{event_type}_antec_mean'] - 
                                    ref_da[f'best_corr_window_{event_type}_antec_mean'])/ref_da.upstream_basin_area
                change_da_driver_median = change_da_driver.quantile(0.5)
                change_da_driver_qmin = change_da_driver.quantile(0.1)
                change_da_driver_q1 = change_da_driver.quantile(0.25)
                change_da_driver_q3 = change_da_driver.quantile(0.75)
                change_da_driver_qmax = change_da_driver.quantile(0.9)
                ref_da_asm_median = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.5)
                ref_da_asm_qmin = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.1)
                ref_da_asm_q1 = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.25)
                ref_da_asm_q3 = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.75)
                ref_da_asm_qmax = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.9)
                ref_da_asm_count = len(ref_da[f'soilm_sat_{soilm_depth}_antec_min'])
                dist_da_asm_median = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.5)
                dist_da_asm_qmin = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.1)
                dist_da_asm_q1 = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.25)
                dist_da_asm_q3 = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.75)
                dist_da_asm_qmax = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.9)
                high_flow_count.append(ref_da_asm_count)
                change_driver_bxpstats.append({
                    metric: val
                    for metric, val in zip(
                        ['med', 'q1', 'q3', 'whislo', 'whishi'],
                        [change_da_driver_median, change_da_driver_q1, change_da_driver_q3, 
                        change_da_driver_qmin, change_da_driver_qmax])})
                ref_asm_bxpstats.append({
                    metric: val
                    for metric, val in zip(
                        ['med', 'q1', 'q3', 'whislo', 'whishi'],
                        [ref_da_asm_median, ref_da_asm_q1, ref_da_asm_q3, 
                        ref_da_asm_qmin, ref_da_asm_qmax])})
                dist_asm_bxpstats.append({
                    metric: val
                    for metric, val in zip(
                        ['med', 'q1', 'q3', 'whislo', 'whishi'],
                        [dist_da_asm_median, dist_da_asm_q1, dist_da_asm_q3, 
                        dist_da_asm_qmin, dist_da_asm_qmax])})
            s = axs[idriver, iregion].bxp(
                bxpstats=ref_asm_bxpstats,
                positions=np.arange(len(ref_asm_bxpstats))-0.3,
                boxprops={'color': 'black'},
                widths=0.2,
                showmeans=False, showfliers=False, shownotches=False)
            s = axs[idriver, iregion].bxp(
                bxpstats=dist_asm_bxpstats,
                positions=np.arange(len(dist_asm_bxpstats))-0.1,
                boxprops={'color': 'red'},
                widths=0.2,
                showmeans=False, showfliers=False, shownotches=False)
            ax2 = axs[idriver, iregion].twinx()  # Create a new y-axis for the bar plots 
            ax2.bar(np.arange(len(high_flow_count)), high_flow_count, 
                    width=0.25, color='black', alpha=0.5, label='Ref Count')
            ax2.set_ylabel('# high flows', loc='bottom')
            ax2.set_ylim([0, 2.2*np.max(high_flow_count)])
            ax2.set_yticks([0, 0.5*np.max(high_flow_count), 1*np.max(high_flow_count)])
            ax2.spines['right'].set_position(('axes', axs[idriver, iregion].spines['left'].get_position()[1]))
            ax2.yaxis.set_ticks_position('left')
            ax2.yaxis.tick_left()
            ax3 = axs[idriver, iregion].twinx()  # Create a new y-axis for the bar plots
            ax3.bxp(
                bxpstats=change_driver_bxpstats,
                positions=np.arange(len(change_driver_bxpstats))+0.1,
                boxprops={'color': 'blue'},
                widths=0.2,
                showmeans=False, showfliers=False, shownotches=False)
            ax3.set_ylabel('abs. change driver [mm day$^{-1}$]', loc='top')
            ax3.set_ylim([-150, 50])
            ax3.set_yticks([-50, 0, 50])
            ax3.hlines(0, -0.5, 2.5, color='black', linewidth=0.5)
            axs[idriver, iregion].set_title(f'{region}, {event_type}')
            axs[idriver, iregion].set_ylim([-1, 1])
            axs[idriver, iregion].set_xlim([-0.5, 2.5])
            axs[idriver, iregion].hlines(0, -0.5, 2.5, color='black', linewidth=0.5)
            axs[idriver, iregion].set_xticks(np.arange(len(ref_asm_bxpstats)))
            axs[idriver, iregion].set_xticklabels(['Only CTRL', 'Only P2K', 'Both'])
            axs[idriver, iregion].set_yticks([0, 0.5, 1])
    [axs[-1, i].set_xlabel(f'High Flow Occurrence') for i in range(3)]
    [axs[i, 0].set_ylabel(f'ASM saturation [-]', loc='top') for i in range(3)]
    plt.tight_layout()
    plt.savefig(
        f'plots/antecedent_conditions/boxplot_high_flow_change_vs_asm_change_vs_driver_change{soilm_depth}_by_driver_mean.png',
        dpi=300)

def replace_nan_by_zero(da):
    return da.where(da.notnull(), 0)

def boxplot_high_flow_change_vs_asm_change_abs(
        ref_data, dist_data, dT, soilm_depth='8cm'):
    comp_event_bool = ((ref_data.best_corr_var == 'ar_precip_melt') | 
                      (ref_data.best_corr_var == 'precip_melt'))
    comp_event_bool |= ((dist_data.best_corr_var == 'ar_precip_melt') | 
                      (dist_data.best_corr_var == 'precip_melt'))
    ref_data = ref_data[~comp_event_bool]
    dist_data = dist_data[~comp_event_bool]
    ref_data_west_coast = ref_data[ref_data['lon'] <= -122]
    dist_data_west_coast = dist_data[dist_data['lon'] <= -122]
    ref_data_mountainous_west = ref_data[(ref_data['lon'] > -122) & (ref_data['lon'] <= -100)]
    dist_data_mountainous_west = dist_data[(dist_data['lon'] > -122) & (dist_data['lon'] <= -100)]
    ref_data_east = ref_data[(ref_data['lon'] > -100) & (ref_data['lon'] <= -67)]
    dist_data_east = dist_data[(dist_data['lon'] > -100) & (dist_data['lon'] <= -67)]
    driver_ylims_list = [[-40, 15], [-40, 15], [-100, 35]]
    driver_yticks_list = [[-15, -10, -5, 0, 5, 10, 15], 
                          [-15, -10, -5, 0, 5, 10, 15], 
                          [-30, -20, -10, 0, 10, 20, 30]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for iregion, (region, ref_data_region, dist_data_region, driver_ylim, driver_yticks) in enumerate(
        [('West Coast', ref_data_west_coast, dist_data_west_coast, driver_ylims_list[0], driver_yticks_list[0]), 
         ('Mountainous West', ref_data_mountainous_west, dist_data_mountainous_west, driver_ylims_list[1], driver_yticks_list[1]), 
         ('East', ref_data_east, dist_data_east, driver_ylims_list[2], driver_yticks_list[2])]):
        event_mask_ref = ref_data_region['is_high_flow'].values
        event_mask_dist = dist_data_region['is_high_flow'].values
        
        ref_da_only_ref_high_flow = df_to_da(
            ref_data_region[event_mask_ref&~event_mask_dist])
        dist_da_only_ref_high_flow = df_to_da(
            dist_data_region[event_mask_ref&~event_mask_dist])
        ref_da_only_dist_high_flow = df_to_da(
            ref_data_region[~event_mask_ref&event_mask_dist])
        dist_da_only_dist_high_flow = df_to_da(
            dist_data_region[~event_mask_ref&event_mask_dist])
        ref_da_ref_dist_high_flow = df_to_da(
            ref_data_region[event_mask_ref&event_mask_dist])
        dist_da_ref_dist_high_flow = df_to_da(
        dist_data_region[event_mask_ref&event_mask_dist])
        ref_asm_bxpstats = []
        dist_asm_bxpstats = []
        change_driver_bxpstats = []
        change_streamflow_bxpstats = []
        high_flow_count_list = []
        event_type_count_list = []
        for ref_da, dist_da, event_type, high_flow_occ_str in zip(
            [ref_da_only_ref_high_flow, ref_da_only_dist_high_flow, ref_da_ref_dist_high_flow],
            [dist_da_only_ref_high_flow, dist_da_only_dist_high_flow, dist_da_ref_dist_high_flow],
            [ref_da_only_ref_high_flow.best_corr_var.values, 
             dist_da_only_dist_high_flow.best_corr_var.values, 
             ref_da_ref_dist_high_flow.best_corr_var.values],
            ["Only CTRL", "Only P2K", "Total"]):
            no_corr_mask = (event_type == 'no_corr')
            event_type = event_type[~no_corr_mask]
            ref_da = ref_da.isel(index=~no_corr_mask)
            dist_da = dist_da.isel(index=~no_corr_mask)
            ref_da_driver_mean = xr.DataArray(
                [ref_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 ref_da[f'zero_shift_window_values_melt_antec_mean'][irow]
                 if et in ['precip', 'ar_precip'] 
                 else ref_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 ref_da[f'zero_shift_window_values_precip_antec_mean'][irow]
                 for irow, et in enumerate(event_type)])
            dist_da_driver_mean = xr.DataArray(
                [dist_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 dist_da[f'zero_shift_window_values_melt_antec_mean'][irow]
                 if et in ['precip', 'ar_precip'] 
                 else dist_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 dist_da[f'zero_shift_window_values_precip_antec_mean'][irow]
                 for irow, et in enumerate(event_type)])
            ref_da_streamflow_mean = xr.DataArray(
                [ref_da[f'high_flow_streamflow'][irow]
                 for irow, et in enumerate(event_type)])
            dist_da_streamflow_mean = xr.DataArray(
                [dist_da[f'high_flow_streamflow'][irow]
                 for irow, et in enumerate(event_type)])
            event_type_count_list.append(
                {t: (event_type == t).sum() 
                 for t in ['ar_precip', 'precip', 'melt']})
            drop_bool = ~(ref_da_driver_mean.isnull() | (ref_da_driver_mean == 0))
            drop_bool &= ~(dist_da_driver_mean.isnull() | (dist_da_driver_mean == 0))
            ref_da_driver_mean = ref_da_driver_mean[drop_bool]
            dist_da_driver_mean = dist_da_driver_mean[drop_bool]
            ref_da_streamflow_mean = ref_da_streamflow_mean[drop_bool]
            dist_da_streamflow_mean = dist_da_streamflow_mean[drop_bool]
            upstream_area = ref_da['upstream_basin_area'][drop_bool.values]
            land_area = ref_da['land_area'][drop_bool.values]
            change_da_driver = (dist_da_driver_mean - ref_da_driver_mean)/upstream_area.values
            change_da_streamflow = (dist_da_streamflow_mean - ref_da_streamflow_mean)/land_area.values
            change_da_driver_median = change_da_driver.quantile(0.5)
            change_da_driver_qmin = change_da_driver.quantile(0.1)
            change_da_driver_q1 = change_da_driver.quantile(0.25)
            change_da_driver_q3 = change_da_driver.quantile(0.75)
            change_da_driver_qmax = change_da_driver.quantile(0.9)
            change_da_streamflow_median = change_da_streamflow.quantile(0.5)
            change_da_streamflow_qmin = change_da_streamflow.quantile(0.1)
            change_da_streamflow_q1 = change_da_streamflow.quantile(0.25)
            change_da_streamflow_q3 = change_da_streamflow.quantile(0.75)
            change_da_streamflow_qmax = change_da_streamflow.quantile(0.9)
            
            ref_da_asm = ref_da[f'soilm_sat_{soilm_depth}_antec_min']
            dist_da_asm = dist_da[f'soilm_sat_{soilm_depth}_antec_min']
            ref_da_asm_median = ref_da_asm.quantile(0.5)
            ref_da_asm_qmin = ref_da_asm.quantile(0.1)
            ref_da_asm_q1 = ref_da_asm.quantile(0.25)
            ref_da_asm_q3 = ref_da_asm.quantile(0.75)
            ref_da_asm_qmax = ref_da_asm.quantile(0.9)
            ref_da_asm_count = len(ref_da_asm)
            dist_da_asm_median = dist_da_asm.quantile(0.5)
            dist_da_asm_qmin = dist_da_asm.quantile(0.1)
            dist_da_asm_q1 = dist_da_asm.quantile(0.25)
            dist_da_asm_q3 = dist_da_asm.quantile(0.75)
            dist_da_asm_qmax = dist_da_asm.quantile(0.9)
            high_flow_count_list.append(ref_da_asm_count)
            change_driver_bxpstats.append({
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [change_da_driver_median, change_da_driver_q1, change_da_driver_q3, 
                    change_da_driver_qmin, change_da_driver_qmax])})
            change_streamflow_bxpstats.append({
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [change_da_streamflow_median, change_da_streamflow_q1, change_da_streamflow_q3, 
                    change_da_streamflow_qmin, change_da_streamflow_qmax])})
            ref_asm_bxpstats.append({
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [ref_da_asm_median, ref_da_asm_q1, ref_da_asm_q3, 
                    ref_da_asm_qmin, ref_da_asm_qmax])})
            dist_asm_bxpstats.append({
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [dist_da_asm_median, dist_da_asm_q1, dist_da_asm_q3, 
                    dist_da_asm_qmin, dist_da_asm_qmax])})
            
        s = axs[iregion].bxp(
            bxpstats=ref_asm_bxpstats,
            positions=np.arange(len(ref_asm_bxpstats))-0.4,
            boxprops={'color': 'black'},
            widths=0.2,
            showmeans=False, showfliers=False, shownotches=False, 
            )
        s = axs[iregion].bxp(
            bxpstats=dist_asm_bxpstats,
            positions=np.arange(len(dist_asm_bxpstats))-0.2,
            boxprops={'color': 'gray'},
            widths=0.2,
            showmeans=False, showfliers=False, shownotches=False)
        ax2 = axs[iregion].twinx()  # Create a new y-axis for the bar plots 
        ax2.spines['right'].set_position(('axes', axs[iregion].spines['left'].get_position()[1]))
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.tick_left()
        ax2.bar(np.arange(len(high_flow_count_list)), high_flow_count_list, 
                width=0.25, color='black', alpha=0.5, label='Ref Count')
        for i, event_type_count in enumerate(event_type_count_list):
            ax2.hlines(event_type_count['ar_precip'], i-0.125, i+0.125, color='blue', linewidth=0.5, label='AR precip')
            ax2.hlines(event_type_count['precip'], i-0.125, i+0.125, color='purple', linewidth=0.5, label='non-AR precip')
            ax2.hlines(event_type_count['melt'], i-0.125, i+0.125, color='red', linewidth=0.5, label='melt')
        # if iregion == 0:
            # ax2.legend(bbox_to_anchor=(0, 0.4))

        ax2.set_ylabel('Count', loc='bottom')  # Move ylabel to the left side
        ax2.yaxis.set_label_position('left')  # Set label position to left
        ax2.set_ylim([0, 2.2*np.max(high_flow_count_list)])
        ax2.set_yticks([0, round(0.4*np.max(high_flow_count_list), -2), round(0.8*np.max(high_flow_count_list), -2)])
        ax3 = axs[iregion].twinx()  # Create a new y-axis for the bar plots
        ax3.bxp(
            bxpstats=change_driver_bxpstats,
            positions=np.arange(len(change_driver_bxpstats))+0.2,
            boxprops={'color': 'blue'},
            widths=0.2,
            showmeans=False, showfliers=False, shownotches=False)
        ax3.bxp(
            bxpstats=change_streamflow_bxpstats,
            positions=np.arange(len(change_driver_bxpstats))+0.4,
            boxprops={'color': 'turquoise'},
            widths=0.2,
            showmeans=False, showfliers=False, shownotches=False)
        ax3.set_ylabel('$\Delta$ driver+streamflow [mm day$^{-1}$]', loc='top')
        ax3.set_ylim(driver_ylim)
        ax3.set_yticks(driver_yticks)
        ax3.hlines(0, -0.5, 2.5, color='black', linewidth=0.5)
        axs[iregion].set_title(f'{region}')
        axs[iregion].set_ylim([-1, 1])
        axs[iregion].set_xlim([-0.5, 2.5])
        axs[iregion].hlines(0, -0.5, 2.5, color='black', linewidth=0.5)
        axs[iregion].set_xticks(np.arange(len(ref_asm_bxpstats)))
        axs[iregion].set_xticklabels(['Only CTRL', 'Only P2K', 'Both'])
        axs[iregion].set_yticks([0, 0.5, 1])
    [axs[i].set_xlabel(f'High Flow Occurrence') for i in range(3)]
    axs[0].set_ylabel(f'ASM saturation [-]', loc='top')
    plt.tight_layout()
    plt.savefig(
        f'plots/antecedent_conditions/boxplot_high_flow_change_vs_asm_change_vs_driver_change{soilm_depth}_abs_homog_driver_windows.png',
        dpi=300)

def boxplot_high_flow_change_vs_asm_change_rel(
        ref_data, dist_data, dT, soilm_depth='8cm'):
    comp_event_bool = ((ref_data.best_corr_var == 'ar_precip_melt') | 
                      (ref_data.best_corr_var == 'precip_melt'))
    comp_event_bool |= ((dist_data.best_corr_var == 'ar_precip_melt') | 
                      (dist_data.best_corr_var == 'precip_melt'))
    ref_data = ref_data[~comp_event_bool]
    dist_data = dist_data[~comp_event_bool]
    ref_data_west_coast = ref_data[ref_data['lon'] <= -122]
    dist_data_west_coast = dist_data[dist_data['lon'] <= -122]
    ref_data_mountainous_west = ref_data[(ref_data['lon'] > -122) & (ref_data['lon'] <= -100)]
    dist_data_mountainous_west = dist_data[(dist_data['lon'] > -122) & (dist_data['lon'] <= -100)]
    ref_data_east = ref_data[(ref_data['lon'] > -100) & (ref_data['lon'] <= -67)]
    dist_data_east = dist_data[(dist_data['lon'] > -100) & (dist_data['lon'] <= -67)]
    driver_ylims_list = [[-70, 30], [-70, 30], [-160, 100]]
    driver_yticks_list = [[-20, 0, 20], [-20, 0, 20], [-30, 0, 30, 60]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for iregion, (region, ref_data_region, dist_data_region, driver_ylim, driver_yticks) in enumerate(
        [('West Coast', ref_data_west_coast, dist_data_west_coast, driver_ylims_list[0], driver_yticks_list[0]), 
         ('Mountainous West', ref_data_mountainous_west, dist_data_mountainous_west, driver_ylims_list[1], driver_yticks_list[1]), 
         ('East', ref_data_east, dist_data_east, driver_ylims_list[2], driver_yticks_list[2])]):
        event_mask_ref = ref_data_region['is_high_flow'].values
        event_mask_dist = dist_data_region['is_high_flow'].values
        
        ref_da_only_ref_high_flow = df_to_da(
            ref_data_region[event_mask_ref&~event_mask_dist])
        dist_da_only_ref_high_flow = df_to_da(
            dist_data_region[event_mask_ref&~event_mask_dist])
        ref_da_only_dist_high_flow = df_to_da(
            ref_data_region[~event_mask_ref&event_mask_dist])
        dist_da_only_dist_high_flow = df_to_da(
            dist_data_region[~event_mask_ref&event_mask_dist])
        ref_da_ref_dist_high_flow = df_to_da(
            ref_data_region[event_mask_ref&event_mask_dist])
        dist_da_ref_dist_high_flow = df_to_da(
        dist_data_region[event_mask_ref&event_mask_dist])
        ref_asm_bxpstats = []
        dist_asm_bxpstats = []
        change_driver_bxpstats = []
        high_flow_count_list = []
        event_type_count_list = []
        for ref_da, dist_da, event_type, high_flow_occ_str in zip(
            [ref_da_only_ref_high_flow, ref_da_only_dist_high_flow, ref_da_ref_dist_high_flow],
            [dist_da_only_ref_high_flow, dist_da_only_dist_high_flow, dist_da_ref_dist_high_flow],
            [ref_da_only_ref_high_flow.best_corr_var.values, 
             dist_da_only_dist_high_flow.best_corr_var.values, 
             ref_da_ref_dist_high_flow.best_corr_var.values],
            ["Only CTRL", "Only P2K", "Total"]):
            no_corr_mask = (event_type == 'no_corr')
            event_type = event_type[~no_corr_mask]
            ref_da = ref_da.isel(index=~no_corr_mask)
            dist_da = dist_da.isel(index=~no_corr_mask)
            ref_da_driver_mean = xr.DataArray(
                [ref_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 ref_da[f'zero_shift_window_values_melt_antec_mean'][irow]
                 if et in ['precip', 'ar_precip'] 
                 else ref_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 ref_da[f'zero_shift_window_values_precip_antec_mean'][irow]
                 for irow, et in enumerate(event_type)])
            dist_da_driver_mean = xr.DataArray(
                [dist_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 dist_da[f'zero_shift_window_values_melt_antec_mean'][irow]
                 if et in ['precip', 'ar_precip'] 
                 else dist_da[f'zero_shift_window_values_{et}_antec_mean'][irow] + 
                 dist_da[f'zero_shift_window_values_precip_antec_mean'][irow]
                 for irow, et in enumerate(event_type)])
            event_type_count_list.append(
                {t: (event_type == t).sum() 
                 for t in ['ar_precip', 'precip', 'melt']})
            drop_bool = ~(ref_da_driver_mean.isnull() | (ref_da_driver_mean == 0))
            drop_bool &= ~(dist_da_driver_mean.isnull() | (dist_da_driver_mean == 0))
            ref_da_driver_mean = ref_da_driver_mean[drop_bool]
            dist_da_driver_mean = dist_da_driver_mean[drop_bool]
            upstream_area = ref_da['upstream_basin_area'][drop_bool.values]
            change_da_driver = (dist_da_driver_mean - ref_da_driver_mean)/ref_da_driver_mean/dT*100
            change_da_driver_median = change_da_driver.quantile(0.5)
            change_da_driver_qmin = change_da_driver.quantile(0.1)
            change_da_driver_q1 = change_da_driver.quantile(0.25)
            change_da_driver_q3 = change_da_driver.quantile(0.75)
            change_da_driver_qmax = change_da_driver.quantile(0.9)
            ref_da_asm_median = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.5)
            ref_da_asm_qmin = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.1)
            ref_da_asm_q1 = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.25)
            ref_da_asm_q3 = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.75)
            ref_da_asm_qmax = ref_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.9)
            ref_da_asm_count = len(ref_da[f'soilm_sat_{soilm_depth}_antec_min'])
            dist_da_asm_median = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.5)
            dist_da_asm_qmin = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.1)
            dist_da_asm_q1 = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.25)
            dist_da_asm_q3 = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.75)
            dist_da_asm_qmax = dist_da[f'soilm_sat_{soilm_depth}_antec_min'].quantile(0.9)
            high_flow_count_list.append(ref_da_asm_count)
            change_driver_bxpstats.append({
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [change_da_driver_median, change_da_driver_q1, change_da_driver_q3, 
                    change_da_driver_qmin, change_da_driver_qmax])})
            ref_asm_bxpstats.append({
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [ref_da_asm_median, ref_da_asm_q1, ref_da_asm_q3, 
                    ref_da_asm_qmin, ref_da_asm_qmax])})
            dist_asm_bxpstats.append({
                metric: val
                for metric, val in zip(
                    ['med', 'q1', 'q3', 'whislo', 'whishi'],
                    [dist_da_asm_median, dist_da_asm_q1, dist_da_asm_q3, 
                    dist_da_asm_qmin, dist_da_asm_qmax])})
        s = axs[iregion].bxp(
            bxpstats=ref_asm_bxpstats,
            positions=np.arange(len(ref_asm_bxpstats))-0.32,
            boxprops={'color': 'black'},
            widths=0.2,
            showmeans=False, showfliers=False, shownotches=False, 
            )
        s = axs[iregion].bxp(
            bxpstats=dist_asm_bxpstats,
            positions=np.arange(len(dist_asm_bxpstats))-0.12,
            boxprops={'color': 'gray'},
            widths=0.2,
            showmeans=False, showfliers=False, shownotches=False)
        ax2 = axs[iregion].twinx()  # Create a new y-axis for the bar plots 
        ax2.spines['right'].set_position(('axes', axs[iregion].spines['left'].get_position()[1]))
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.tick_left()
        ax2.bar(np.arange(len(high_flow_count_list)), high_flow_count_list, 
                width=0.25, color='black', alpha=0.5, label='Ref Count')
        for i, event_type_count in enumerate(event_type_count_list):
            ax2.hlines(event_type_count['ar_precip'], i-0.125, i+0.125, color='blue', linewidth=0.5, label='AR precip')
            ax2.hlines(event_type_count['precip'], i-0.125, i+0.125, color='purple', linewidth=0.5, label='non-AR precip')
            ax2.hlines(event_type_count['melt'], i-0.125, i+0.125, color='red', linewidth=0.5, label='melt')
        # if iregion == 0:
            # ax2.legend(bbox_to_anchor=(0, 0.4))

        ax2.set_ylabel('Count', loc='bottom')  # Move ylabel to the left side
        ax2.yaxis.set_label_position('left')  # Set label position to left
        ax2.set_ylim([0, 2.2*np.max(high_flow_count_list)])
        ax2.set_yticks([0, round(0.4*np.max(high_flow_count_list), -2), round(0.8*np.max(high_flow_count_list), -2)])
        ax3 = axs[iregion].twinx()  # Create a new y-axis for the bar plots
        ax3.bxp(
            bxpstats=change_driver_bxpstats,
            positions=np.arange(len(change_driver_bxpstats))+0.12,
            boxprops={'color': 'blue'},
            widths=0.2,
            showmeans=False, showfliers=False, shownotches=False)
        ax3.set_ylabel('abs. change driver [mm day$^{-1}$]', loc='top')
        ax3.set_ylim(driver_ylim)
        ax3.set_yticks(driver_yticks)
        ax3.hlines(0, -0.5, 2.5, color='black', linewidth=0.5)
        axs[iregion].set_title(f'{region}')
        axs[iregion].set_ylim([-1, 1])
        axs[iregion].set_xlim([-0.5, 2.5])
        axs[iregion].hlines(0, -0.5, 2.5, color='black', linewidth=0.5)
        axs[iregion].set_xticks(np.arange(len(ref_asm_bxpstats)))
        axs[iregion].set_xticklabels(['Only CTRL', 'Only P2K', 'Both'])
        axs[iregion].set_yticks([0, 0.5, 1])
    [axs[i].set_xlabel(f'High Flow Occurrence') for i in range(3)]
    axs[0].set_ylabel(f'ASM saturation [-]', loc='top')
    plt.tight_layout()
    plt.savefig(
        f'plots/antecedent_conditions/boxplot_high_flow_change_vs_asm_change_vs_driver_change{soilm_depth}_rel_homog_driver_windows.png',
        dpi=300)

def merge_to_unique_high_flow_events(df1_is_high_flow, df2_might_be_high_flow):
    # Mask to drop events in df1 that are also in df2
    df1_in_df2_mask = np.where(
        pd.merge(
            df1_is_high_flow, df2_might_be_high_flow, 
            on=['time', 'lat', 'lon'], how='left', 
            indicator='Exist').Exist == 'both', 
        True, False)
    # Mask to know which high flows in df2 are actually high-flows 
    # by comparing which high flows in df2 are also in df1
    df2_is_high_flow_mask = np.where(
        pd.merge(
            df2_might_be_high_flow, df1_is_high_flow, 
            on=['time', 'lat', 'lon'], how='left', 
            indicator='Exist').Exist == 'both', 
        True, False)
    # Drop events in df1 that are also in df2
    df1_is_high_flow = df1_is_high_flow[~df1_in_df2_mask]
    # Set is_high_flow to True for df1
    df1_is_high_flow['is_high_flow'] = True
    # Set is_high_flow to False for df2
    df2_might_be_high_flow['is_high_flow'] = False
    # Set is_high_flow to True for df2 where it is actually a high-flow
    df2_might_be_high_flow['is_high_flow'][df2_is_high_flow_mask] = True
    combined_df = pd.concat([df1_is_high_flow, df2_might_be_high_flow])
    return combined_df

def get_land_area(df, land_area):
    return float(land_area.sel(lon=df.lon, lat=df.lat, method='nearest').values)

def _main():
    exp_name_ctrl = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min'
    exp_name_p2K = 'c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K'
    start_year = 1951
    end_year = 2018
    huc_level = 2
    dT = get_global_mean_dT(exp_name_ctrl, exp_name_p2K, start_year, end_year)
    land_area = lon_360_to_180(xr.open_dataset(
        f'/archive/Ming.Zhao/awg/2023.04/{exp_name_ctrl}/'
        'gfdl.ncrc5-intel23-classic-prod-openmp/pp/land/land.static.nc')['land_area'])
    high_flow_data_ctrl = load_high_flow_data(
        exp_name_ctrl, start_year, end_year, 
        specific_high_flow_mask=None)
    high_flow_data_ctrl_p2K_mask = load_high_flow_data(
        exp_name_ctrl, start_year, end_year, 
        specific_high_flow_mask='p2K_high_flow_mask')
    high_flow_data_p2K = load_high_flow_data(
        exp_name_p2K, start_year, end_year, 
        specific_high_flow_mask=None)
    high_flow_data_p2K_ctrl_mask = load_high_flow_data(
        exp_name_p2K, start_year, end_year, 
        specific_high_flow_mask='ctrl_high_flow_mask')
    high_flow_data_ctrl = merge_to_unique_high_flow_events(
        high_flow_data_ctrl, high_flow_data_ctrl_p2K_mask)
    high_flow_data_p2K = merge_to_unique_high_flow_events(
        high_flow_data_p2K, high_flow_data_p2K_ctrl_mask)
    high_flow_data_ctrl, high_flow_data_p2K = match_time_lat_lon_order(
        high_flow_data_ctrl, high_flow_data_p2K)
    high_flow_data_ctrl['land_area'] = high_flow_data_ctrl.apply(
        get_land_area, axis=1, args=(land_area, ))
    high_flow_data_p2K['land_area'] = high_flow_data_p2K.apply(
        get_land_area, axis=1, args=(land_area, ))
    boxplot_high_flow_change_vs_asm_change_abs(
        high_flow_data_ctrl, high_flow_data_p2K, dT, soilm_depth='1cm')

if __name__ == '__main__':
    _main()
