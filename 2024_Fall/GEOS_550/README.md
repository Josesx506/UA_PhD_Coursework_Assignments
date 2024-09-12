### Numerical Modeling of Geomorphological Processes
1. Compile c programs with `gcc` e.g. `gcc -o <output_file> <input_script.c>`
2. Run the program with `./<output_file>`. The output file can work with or without a file name extension.
```C
>$ ./hello 
Hello World
```

### Running the lem2d code
> [!Note]: Using ***gcc***, the source code should be compiled using `gcc -g -o lem2d.o lem2d.c util.c userdefinedfunctions.c`. A `/movie` directory must 
be created in the local directory in which *lem2d* is run.<br>

Additional scripts I created are
1. `create_mask.py` - Create a simple fault mask with python
2. `run_lem2d.sh` - Run the lem2d program and create movie folder automatically
3. `convert_xyz_to_tiff.sh` - Loop gdal to convert the xyz files to tif for paraview

Configuration can be defined in the `lem2dpackage/input.txt`. The current grid dimensions are 101 by 101 cells, where each cell has a node spacing of 30 m, creating a *3 by 3 km* model grid.

- The fault mask can be created using the pyton script `python create_mask.py 101 101`, and matching the input rows and columns.
- Create the movie folder if it doesn't exist and run the lem2d program with `./lem2d.o`
- Convert the `.xyz` files to `.tif` by running the `./convert_xyz_to_tiff.sh movie` command from the lem2d parent directory where the *movie* directory was created. This uses **gdal** to convert the files, so gdal must be installed for it to work. `brew install gdal` if you don't have it.
- Import the `.tif*` into paraview which is a stack of tif files and don't load each file individually. The **`*`** indicates it's a stack of images that can be animated.
- Right-click on the image-stack and uncheck the `Read as Image Stack` option, then  apply the `Warp by Scalar` filter to the image stack.
- Adjust the view and window size in the layout window, then click on `File -> Save Animation` to export the animation as a video.