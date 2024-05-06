import xarray as xr
import multiprocessing
import numpy as np
import argparse

import data_util as arp

parser = argparse.ArgumentParser()
parser.add_argument("--lon_min", type=float,)
parser.add_argument("--lon_max", type=float,)
parser.add_argument("--exp_name", type=str)
args = parser.parse_args()


na_data = xr.open_mfdataset(
    f'/archive/Marc.Prange/na_data/{args.exp_name}/{args.exp_name}_na_*.nc').load()

arp.store_na_temporal_comp_for_lon_range(
    na_data, args.lon_min, args.lon_max, 
    exp_name=args.exp_name,
    min_precip=20,
    precip_var='pr',
    ar_day=False,
    winter=True)