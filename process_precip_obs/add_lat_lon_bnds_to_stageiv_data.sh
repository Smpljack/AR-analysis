#!/bin/bash

module load nco

for year in 2001 ; 
do
for mon in Mar ; 
do
    file="/archive/m2p/StageIV/daily/StageIV_${mon}_${year}.daily.nc"
    outfile="/archive/m2p/StageIV/daily/StageIV_${mon}_${year}.daily.bnds.nc"
    ncap2 -O -s '        
    lat_bnds[y,x,bnds]=double(0);
    lon_bnds[y,x,bnds]=double(0);
    // Compute latitude bounds
    do i=0; i<y; i++;
      do j=0; j<x; j++;
        if (i > 0 && i < y-1) then
          lat_bnds(i,j,0) = lat(i,j) - (lat(i,j) - lat(i-1,j)) / 2;
          lat_bnds(i,j,1) = lat(i,j) + (lat(i+1,j) - lat(i,j)) / 2;
        elseif (i == 0) then
          lat_bnds(i,j,0) = lat(i,j) - (lat(i+1,j) - lat(i,j)) / 2;
          lat_bnds(i,j,1) = lat(i,j) + (lat(i+1,j) - lat(i,j)) / 2;
        else
          lat_bnds(i,j,0) = lat(i,j) - (lat(i,j) - lat(i-1,j)) / 2;
          lat_bnds(i,j,1) = lat(i,j) + (lat(i,j) - lat(i-1,j)) / 2;
        endif
      end do
    end do
    
    // Compute longitude bounds
    do i=0; i<y; i++;
      do j=0; j<x; j++;
        if (j > 0 && j < x-1) then
          lon_bnds(i,j,0) = lon(i,j) - (lon(i,j) - lon(i,j-1)) / 2;
          lon_bnds(i,j,1) = lon(i,j) + (lon(i,j+1) - lon(i,j)) / 2;
        elseif (j == 0) then
          lon_bnds(i,j,0) = lon(i,j) - (lon(i,j+1) - lon(i,j)) / 2;
          lon_bnds(i,j,1) = lon(i,j) + (lon(i,j+1) - lon(i,j)) / 2;
        else
          lon_bnds(i,j,0) = lon(i,j) - (lon(i,j) - lon(i,j-1)) / 2;
          lon_bnds(i,j,1) = lon(i,j) + (lon(i,j) - lon(i,j-1)) / 2;
        endif
      end do
    end do
    ' ${file} ${outfile}
    # Add longitude attributes
    ncatted -a long_name,lon_bnds,o,c,"longitude bounds" ${outfile}
    ncatted -a units,lon_bnds,o,c,"degrees_east" ${outfile}
    ncatted -a _CoordinateAxisType,lon_bnds,o,c,"Lon" ${outfile}
    # Add latitude attributes
    ncatted -a long_name,lat_bnds,o,c,"latitude bounds" ${outfile}
    ncatted -a units,lat_bnds,o,c,"degrees_north" ${outfile}
    ncatted -a _CoordinateAxisType,lat_bnds,o,c,"Lat" ${outfile}
    # Add bnds as attributes to lat and lon
    ncatted -a bounds,lat,o,c,"lat_bnds" ${outfile}
    ncatted -a bounds,lon,o,c,"lon_bnds" ${outfile}
    
    # Rename dimensions x to longitude and y to latitude
    # ncrename -d x,longitude -d y,latitude ${outfile}
done
done
