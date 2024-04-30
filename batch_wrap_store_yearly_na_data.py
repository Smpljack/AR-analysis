import os
import numpy as np
import xarray as xr

for year in range(1970, 1990):
    os.system("sbatch "
              "/home/Marc.Prange/work/AR-analysis/batch_wrapper.sh "
             f"{year}")
