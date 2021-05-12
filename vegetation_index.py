import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import rasterio as rs

# img_loc = "/home/sytze/Desktop/tile_pinet1-2019_512_4096.jpg"
# img = Image.open(img_loc)
# img_array = np.array(img)

def calculate_RGBVI(image_fpath):
    np.seterr(divide='ignore', invalid='ignore')

    with rs.open(image_fpath) as src:
        R = src.read(1).astype(float)
        G = src.read(2).astype(float)
        B = src.read(3).astype(float)

    RGBVI = (G - B * R) / ((G)**2 + (B * R))

    return RGBVI

#result = calculate_RGBVI(img_loc)

#plt.imshow(img_array)
#plt.imshow(result, alpha=0.5, cmap="viridis")
#plt.show()