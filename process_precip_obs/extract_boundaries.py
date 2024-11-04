import sys
import numpy as np
from netCDF4 import Dataset

if len(sys.argv) != 2:
    print("Usage: python extract_boundaries.py <stageiv_file>")
    sys.exit(1)

stageiv_file = sys.argv[1]

try:
    ds = Dataset(stageiv_file, 'r')
except IOError:
    print(f"Error: Cannot open {stageiv_file}")
    sys.exit(1)

lon = ds.variables['lon'][:]  # Adjust variable name if different
lat = ds.variables['lat'][:]  # Adjust variable name if different

west = np.min(lon)
east = np.max(lon)
south = np.min(lat)
north = np.max(lat)

print(f"{west} {east} {south} {north}")

ds.close()