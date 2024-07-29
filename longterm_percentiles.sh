#!/bin/bash

module load cdo

# Define the range of years to merge
start_year=1980
end_year=2019
# Percentile to calculate
pctl=95
# Experiment and path
exp='c192L33_am4p0_amip_HIRESMIP_nudge_wind_1day_p2K'
path="/archive/Marc.Prange/na_data/${exp}"
input_pattern="${path}/${exp}_na_YYYY.nc"
output_file_merged="${path}/${exp}_na_${start_year}-${end_year}_merged_file.nc"
output_file_pctl="${path}/${exp}_na_${start_year}-${end_year}_pctl${pctl}.nc"

# Initialize an array to store file paths
file_paths=()

# Loop through each year to generate file paths
for (( year=$start_year; year<=$end_year; year++ )); do
    # Replace YYYY in input pattern with current year
    current_file=$(echo $input_pattern | sed "s/YYYY/$year/g")
    
    # Check if the file exists
    if [ -f "$current_file" ]; then
        file_paths+=("$current_file")
    else
        echo "Warning: File $current_file not found."
    fi
done

# Print the list of file paths
echo "List of existing yearly files:"
printf "%s\n" "${file_paths[@]}"

# Merge all yearly files into one output file
cdo -s -O mergetime "${file_paths[@]}" "$output_file_merged"

# Calculate 95th percentile over time dimension
cdo -s -O timpctl,$pctl "$output_file_merged"  -timmin "$output_file_merged" -timmax "$output_file_merged" "$output_file_pctl"

# Clean up temporary files if needed
# rm "$output_file_merged"

echo "Processing complete."
