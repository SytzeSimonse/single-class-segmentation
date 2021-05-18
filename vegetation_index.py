import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import rasterio as rs
from rasterio.windows import shape

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

    dataset = rs.open(image_fpath)

    img_width = dataset.width
    img_height = dataset.height
    
    R = dataset.read(1).astype(float)
    G = dataset.read(2).astype(float)
    B = dataset.read(3).astype(float)

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

    # with rs.open(image_fpath) as src:
    #     img_width = src.width
    #     img_height = src.height

    #     R = src.read(1).astype(float)
    #     G = src.read(2).astype(float)
    #     B = src.read(3).astype(float)

    #     profile = src.profile

    # # Calculating VI
    # if VI == 'ExG':
    #     VI_result = calculate_ExG(R, G, B)
    # elif VI == 'RGBVI':
    #     VI_result = calculate_RGBVI(R, G, B)

    # result = np.zeros((img_height, img_width))
    # result[VI_result > 0.2] = 1

    # plt.imshow(result)
    # plt.show()

    # with rs.open(output_fpath, 'w', **profile) as dst:
    #     profile = tile_dest.profile

    #     profile.update(
    #         nodata=0,
    #         dtype=rs.uint8,
    #         count=1,
    #         compress='LZW')

    #     dst.write(result.astype(rs.uint8), 1)

img_loc = "/home/sytze/example.tif"
mask = calculate_VI(img_loc, output_fpath="my_mask.tif")

# final = np.zeros((img_height, img_width))
# final[result>0.2] = 1

# plt.imshow(mask)
# plt.show()