#!/bin/bash
clear

# Define the folder name
folder="movie"

# Check if the folder exists
if [ ! -d "$folder" ]; then
    echo "Folder '$folder' does not exist. Creating it..."
    mkdir "$folder"
    echo "Folder '$folder' created."
else
    echo "Folder '$folder' already exists, cleaning house."
    rm -rf "$folder"
    mkdir "$folder"
fi

# Run the program
echo "Running the program ./lem2d.o"
./lem2d.o

# Check if the program executed successfully
if [ $? -eq 0 ]; then
    echo "Program executed successfully."
else
    echo "Program execution failed."
fi
