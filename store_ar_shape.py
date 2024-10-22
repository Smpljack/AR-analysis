import xarray as xr
import os
import glob

def preprocess_ds(ds):
    return ds.drop_vars(['islnd', 'iscst', 'axis', 'lfloc'])

def process_ar_data(base_path, exp_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Scan for available files and extract years
    file_pattern = f'{base_path}/{exp_name}/AR_climlmt/{exp_name}_AR_*.nc'
    available_files = glob.glob(file_pattern)
    available_years = [int(file.split('_')[-1].split('.')[0]) for file in available_files]
    
    if not available_years:
        print(f"No files found for experiment {exp_name}")
        return

    start_year, end_year = min(available_years), max(available_years)
    print(f"Processing years from {start_year} to {end_year}")

    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}")
        
        # Load AR data for the specific year
        ar_data = xr.open_mfdataset(
            f'{base_path}/{exp_name}/AR_climlmt/{exp_name}_AR_{year}.nc',
            parallel=False, concat_dim="time", combine="nested",
            data_vars='minimal', coords='minimal', compat='override', preprocess=preprocess_ds
        )

        # Resample to daily frequency and store AR shape
        daily_ar_shape = ar_data.shape.resample(time='D').sum() > 0
        
        # Save the daily AR shape data for the year
        output_file = f'{output_dir}/{exp_name}_daily_AR_shape_{year}.nc'
        daily_ar_shape.to_netcdf(output_file)
        
        print(f"Saved daily AR shape data for year {year} to {output_file}")

        # Close the dataset to free up memory
        ar_data.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process AR data for a given experiment.')
    parser.add_argument('--exp_name', type=str, help='Experiment name', 
                        default='c192L33_am4p0_amip_HIRESMIP_HX')
    args = parser.parse_args()
    base_path = '/archive/Ming.Zhao/awg/2022.03/'
    output_dir = f'/archive/Marc.Prange/ar_shape/{args.exp_name}'
    process_ar_data(base_path, args.exp_name, output_dir)
