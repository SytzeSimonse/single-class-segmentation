from PIL import Image
from pathlib import Path
from osgeo import gdal
import subprocess
import os
import shutil
import numpy as np
from tqdm import tqdm # progress bar
import rasterio as rs

def tile_ortomosaic(ortomosaic_fname: str, output_folder: str, tile_size: int = 512, overwrite: bool = False):
    """Splits ortomosaic into smaller tiles.    

    Args:
        ortomosaic_fname (str): Filename of ortomosaic.
        output_folder (str): Folder to store the tiles.
        tile_size (int, optional): Size of tile in pixels. Defaults to 512.
    """
    # Creating paths
    ortomosaic_file_path = Path(ortomosaic_fname)
    output_folder_path = Path(output_folder)

    # Removing output folder (if already exists)
    if output_folder_path.exists() and overwrite:
        shutil.rmtree(output_folder_path)
        # Creating folder for output
        os.mkdir(output_folder_path)
    # Printing message if output folder exists, but should not be overwritten    
    elif output_folder_path.exists() and not overwrite:
        print(f"The folder {str(output_folder_path)} already exists and will not be overwritten.")
    else:
        os.mkdir(output_folder_path)

    # Checking if ortomosaic exists
    if not ortomosaic_file_path.exists():
        assert IOError(
            f"The file '{str(ortomosaic_file_path)} does not exist."
        )

    # Opening ortomosaic
    ortomosaic = gdal.Open(str(ortomosaic_file_path))

    # Selecting first band (= trivial)
    band = ortomosaic.GetRasterBand(1)

    # Getting width and height of ortomosaic
    xsize = band.XSize
    ysize = band.YSize

    # Creating prefix for tile filenames
    output_filename_prefix = "/tile_"

    # Creating tiles 
    for i in tqdm(range(0, xsize, tile_size), desc=f"Tiling..."):
        for j in range(0, ysize, tile_size):
            com_string = f"gdal_translate -of JPEG -srcwin {i}, {j}, {tile_size}, {tile_size} {ortomosaic_file_path} {output_folder_path}{output_filename_prefix}{i}_{j}.tif -co COMPRESS=JPEG"
            subprocess.run(com_string.split(),  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def create_mosaic(input_folder: str):
    """Creates a mosaic from .TIFF files in folder.

    Args:
        input_folder (str): Path to folder.
    """
    # Getting all files in input folder
    tiles_to_mosaic = os.listdir(input_folder)

    # Creating a .txt file with a list of .TIFF files
    files_to_txt_list(input_folder)

    # Creating a string to build a .vrt file from a list of .tif files
    gdal_build_vrt_str = f"gdalbuildvrt -input_file_list tif_files.txt -b 1 output.vrt"
    subprocess.run(gdal_build_vrt_str.split()) # RUN

    # Creating a string for converting .vrt file to .tif file
    gdal_translate_str = f"gdal_translate output.vrt result.tif"
    subprocess.run(gdal_translate_str.split()) # RUN
    
    # Removing .vrt file
    os.remove("output.vrt")

import re

# https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
def sorted_alphanumeric(data: list):
    """Sorts list of items.

    Args:
        data (list): List of items.

    Returns:
        list: Sorted list of items.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def files_to_txt_list(input_folder: str, output_name: str = "tif_files.txt"):
    """Creates a .txt file with file names separated by lines.

    Args:
        input_folder (str): Input folder.

    Example:
    files = ['A.tif', 'B.tif', ..., 'N.tif']
    -> TXT file:
    A.tif
    B.tif
    ...
    C.tif
    """
    # Raising error if output filename is not .txt
    if not output_name.endswith(".txt"):
        raise IOError(
            f"'{output_name}' should end with '.txt'."
        )

    # Getting files from input folder and sorting them 
    files = sorted_alphanumeric(os.listdir(input_folder))
    
    # Writing filenames to .txt file
    with open(output_name, 'w') as f:
        for file in files:
            if file.endswith(".tif"):
                f.write(f"{input_folder}/{file}\n")

def calculate_VI(img_path):
    np.seterr(divide='ignore', invalid='ignore')

    img = Image.open(img_path)
    img = np.array(img)

    R = img[:,:,0] 
    G = img[:,:,1] 
    B = img[:,:,2] 

    result = G
    result[G<200] = 0
    result[G>=200] = 255

    return result

def make_inferences(tiles_folder: str, output_folder: str, inference_function, verbose: bool = False):
    # Creating path for output folder
    output_folder_path = Path(output_folder)
    tiles_folder_path = Path(tiles_folder)

    # Checking if tiles folder exists
    if not tiles_folder_path.exists():
        raise IOError(
            f"{tiles_folder} does not exist."
        )

    # Creating folder for output
    if not output_folder_path.exists():
        os.mkdir(output_folder_path)

    # Getting all tiles
    tiles = os.listdir(tiles_folder)
    
    # Looping through tiles
    for tile in tiles:
        # Checking if tile ends with .tif (and not .aux.xml)
        if tile.endswith(".tif"):
            # Creating full path to tile
            tile_path = tiles_folder_path / tile

            # Print filename
            if verbose:
                print(tile_path)

            # Opening tile with RasterIO
            tile_dest = rs.open(tile_path)

            # Making inference
            img = Image.open(tile_path)
            img = np.array(img)

            inference = inference_function(img)

            # Saving name
            inference_fname = output_folder_path / tile

            # Register GDAL format drivers and configuration options with a
            # context manager.
            with rs.Env():
                # Write an array as a raster band to a new 8-bit file. For
                # the new file's profile, we start with the profile of the source
                profile = tile_dest.profile

                # And then change the band count to 1, set the
                # dtype to uint8, and specify LZW compression.
                profile.update(
                    nodata=0,
                    dtype=rs.ubyte,
                    nbits=1,
                    count=1,
                    compress='CCITRLE')

                # Storing .tif image in original CRS
                with rs.open(inference_fname, 'w', **profile) as dst:
                    dst.write(inference.astype(rs.uint8), 1)
