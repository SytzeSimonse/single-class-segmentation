import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import rasterio as rs
from rasterio.windows import shape
import os

def calculate_RGBVI(R, G, B):
    return (G - B * R) / ((G)**2 + (B * R))

def calculate_ExG(R, G, B):
    return (2*G-R-B) / (G+R+B)

def calculate_VI(image_fpath: str, output_fpath: str, VI: str = 'ExG'):
    # Checking if VI is available
    if VI not in ['RGBVI', 'ExG']:
        raise ValueError(
            f"{VI} is not available."
        )

    # Allowing division by zero and invalid
    np.seterr(divide='ignore', invalid='ignore')

    print("DO WE GET HERE?")

    dataset = rs.open(image_fpath)

    print("... AND HERE?")

    img_width = dataset.width
    img_height = dataset.height

    print("... AND ALSO HERE?")
    
    R = dataset.read(1).astype(float)

    print("DO WE READ THIS?")

    G = dataset.read(2).astype(float)

    print("DO WE READ G?")
    
    B = dataset.read(3).astype(float)

    print("SUCCESS!")

    # Calculating VI
    if VI == 'ExG':
        VI_result = calculate_ExG(R, G, B)
    elif VI == 'RGBVI':
        VI_result = calculate_RGBVI(R, G, B)

    result = np.zeros((img_height, img_width))
    result[VI_result > 0.1] = 1

    with rs.Env():
        profile = dataset.profile

        profile.update(
            nodata=0,
            dtype=rs.uint8,
            count=1,
            compress='LZW')

        # Storing .TIF image in original CRS
        with rs.open(output_fpath, 'w', **profile) as dst:
            dst.write(result.astype(rs.uint8), 1)

def extract_VI_from_ortomosaic(fpath: str, output_folder: str):
    fname = os.path.splitext(fpath)[0]
    output_fpath = fname + "_VI.tif"
    
    mask = calculate_VI(fpath, output_fpath=output_fpath)