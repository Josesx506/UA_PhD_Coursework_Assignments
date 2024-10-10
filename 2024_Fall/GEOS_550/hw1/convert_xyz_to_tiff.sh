#!/bin/bash

# Check if a directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <folder-path>"
    exit 1
fi

# Assign the first argument (the folder path) to a variable
folder="$1"

# Check if the folder exists
if [ ! -d "$folder" ]; then
    echo "Error: Folder '$folder' does not exist."
    exit 1
fi

# Loop through all .xyz files in the specified folder
for infile in "$folder"/*.xyz; do
    # Replace the .xyz extension with .tif for the output file
    outfile="${infile%.xyz}.tif"
    
    # Run gdal_translate to convert the file and save in the same folder
    gdal_translate -ot Float32 -of GTiff "$infile" "$outfile"
    
    echo "Converted $infile to $outfile"
done
