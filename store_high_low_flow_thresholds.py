import xarray as xr
import numpy as np

import data_util as du

def store_low_high_flow_thresholds(
    exp_name, base_path, out_path, start_year, end_year, percentiles=[0.1, 0.9]):
    discharge_long = xr.concat(
    [
        du.load_model_data(
            base_path='/archive/Ming.Zhao/awg/2022.03/',
            year=year,
            variables=['ts', 'rv_o_h2o'],
            exp_name='c192L33_am4p0_amip_HIRESMIP_HX',
            gfdl_processor='gfdl.ncrc4-intel-prod-openmp',
            ar_analysis=False) 
        for year in range(start_year, end_year+1)
        ], dim='time')

def _main():
    pass

if __name__ == '__main__':
    _main()