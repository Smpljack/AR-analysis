import sys
import xarray as xr

def sort_longitudes(input_file, output_file):
    # Open the dataset
    ds = xr.open_dataset(input_file)
    lon_attrs = ds.lon.attrs.copy()  # Make a copy of the attributes 
    # Shift and sort longitudes
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    ds = ds.sortby('lon')
    ds.lon.attrs = lon_attrs
    # Save the sorted dataset
    ds.to_netcdf(output_file)
    print(f"Sorted longitudes and saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sort_longitudes.py <input_file> <output_file>")
        sys.exit(1)
    
    input_nc = sys.argv[1]
    output_nc = sys.argv[2]
    
    sort_longitudes(input_nc, output_nc)