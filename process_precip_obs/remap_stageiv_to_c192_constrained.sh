#!/bin/bash
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=gfdl_w
#SBATCH --job-name=stageiv_remap
#SBATCH --chdir=.
#SBATCH -o /home/Marc.Prange/work/AR-analysis/logs/%x.o%j
#SBATCH -e /home/Marc.Prange/work/AR-analysis/logs/%x.e%j

module load cdo
module load nco

# Check if both year and month are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <year> <month>"
    exit 1
fi

year=$1
mon=$2

echo "Running remap for Year: $year, Month: $mon"

# Input files
stageiv_file="/archive/m2p/StageIV/daily/StageIV_${mon}_${year}.daily.bnds.nc"
mswep_grid="/archive/Ming.Zhao/awg/2022.03/c192_obs/atmos_data/daily_mswep/mswep.19810101-19811231.precipitation.nc"

# Verify StageIV file exists
if [ ! -f "$stageiv_file" ]; then
    echo "Error: StageIV file does not exist at $stageiv_file"
    exit 1
fi

# Extract longitude and latitude data using ncdump
echo "Extracting longitude and latitude data from StageIV file..."

# Extract boundaries using Python
boundaries=$(python extract_boundaries.py "$stageiv_file")
read west east south north <<< $boundaries

# Debug: Print extracted boundaries
echo "Extracted Boundaries:"
echo "West: $west"
echo "East: $east"
echo "South: $south"
echo "North: $north"

# Validate extracted boundaries
if [ -z "$west" ] || [ -z "$east" ] || [ -z "$south" ] || [ -z "$north" ]; then
    echo "Error: Failed to extract boundary coordinates."
    exit 1
fi

# Shift longitude of mswep_grid from 0-360 to -180-180 using ncap2
echo "Shifting longitude of MSWEP grid from 0-360 to -180-180..."

mswep_grid_shifted="mswep_shifted_${mon}_${year}.nc"

ncap2 -O -s 'where(lon > 180) lon=lon-360' "$mswep_grid" "$mswep_grid_shifted"
if [ $? -ne 0 ]; then
    echo "Error: Failed to shift longitude of MSWEP grid using ncap2."
    exit 1
fi

# Sort the longitudes in ascending order using Python's xarray
echo "Sorting longitudes in ascending order using Python..."
mswep_grid_sorted="mswep_sorted_${mon}_${year}.nc"

python sort_longitudes.py "$mswep_grid_shifted" "$mswep_grid_sorted"
if [ $? -ne 0 ]; then
    echo "Error: Failed to sort longitudes using Python."
    # Clean up shifted grid before exiting
    rm "$mswep_grid_shifted"
    exit 1
fi

# Define output file
outfile="/archive/Marc.Prange/StageIV/daily/StageIV_${mon}_${year}.c192.con.daily.nc"
temp_constrained="temp_constrained.nc"

# Select the boundary area from mswep_grid based on StageIV file boundaries
echo "Selecting the constrained area from MSWEP grid..."
cdo sellonlatbox,$west,$east,$south,$north "$mswep_grid_sorted" "$temp_constrained"
if [ $? -ne 0 ]; then
    echo "Error: cdo sellonlatbox command failed."
    exit 1
fi

cdo griddes ${temp_constrained} > griddes.grd

# Remap StageIV data to the constrained MSWEP grid
echo "Remapping StageIV data to the constrained MSWEP grid..."
cdo remapbil,griddes.grd "$stageiv_file" "$outfile"
if [ $? -ne 0 ]; then
    echo "Error: cdo remapcon command failed."
    exit 1
fi

echo "Remapping completed successfully. Output file: $outfile"

# Clean up temporary files
# rm "$temp_constrained"
