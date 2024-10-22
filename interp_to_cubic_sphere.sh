#!/bin/bash
module load cdo

# Set input and output directories
EXP_NAME="c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min_p2K"
INPUT_DIR="/archive/m2p/ts_all_missing_vars/${EXP_NAME}/ts_all"
GRID_DIR="/archive/Ming.Zhao/awg/2023.04/c192L33_am4p0_amip_HIRESMIP_nudge_wind_30min/gfdl.ncrc5-intel23-classic-prod-openmp/pp/land_cubic"
OUTPUT_DIR="/archive/Marc.Prange/ts_all_missing_vars/${EXP_NAME}/ts_all"
VAR_NAME="ar_pr"

# Loop through tiles 3 and 5
for TILE in 3 5; do
    # Set grid file for the current tile
    GRID_FILE="${GRID_DIR}/land_cubic.static.tile${TILE}.nc"
    echo $GRID_FILE
    # Loop through all ar_precip files in the input directory
    for INPUT_FILE in "${INPUT_DIR}/atmos_cmip.195101-202012.${VAR_NAME}.nc"; do
        # Get the filename without path
        FILENAME=$(basename "$INPUT_FILE")
        
        # Set output file name
        OUTPUT_FILE="${OUTPUT_DIR}/${FILENAME%.nc}.tile${TILE}.nc"

        # Perform regridding using CDO
        cdo -O remapbil,${GRID_FILE} -selname,${VAR_NAME} ${INPUT_FILE} ${OUTPUT_FILE}

        echo "Regridded ${FILENAME} to tile ${TILE}"
    done
done

echo "Regridding complete for tiles 3 and 5"
