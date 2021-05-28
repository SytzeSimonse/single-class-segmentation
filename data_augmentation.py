# This module is used to create synthetic data 
# by using the data augmentation functions from 
# the 'transformation_functions.py' module.

import os
from PIL import Image
from matplotlib import pyplot as plt
import imageio

from transformation_functions import random_rotation

#/home/sytze/Code/DeepLabV3Plus_implementation-single-class/tiles

img_dir = "/home/sytze/Code/DeepLabV3Plus_implementation-single-class/tiles/Images"
mask_dir = "/home/sytze/Code/DeepLabV3Plus_implementation-single-class/tiles/Masks"
output_img_dir = "/home/sytze/Code/DeepLabV3Plus_implementation-single-class/tiles_augmented/Images"
output_mask_dir = "/home/sytze/Code/DeepLabV3Plus_implementation-single-class/tiles_augmented/Masks"

tiles = os.listdir(img_dir)

for tile in tiles:
    img = Image.open(os.path.join(img_dir, tile))
    tile_png = os.path.splitext(tile)[0] + ".png"
    mask = Image.open(os.path.join(mask_dir, tile_png))

    pair = {"image": img, "mask": mask}

    transformed = random_rotation(pair)

    tile_img_name = os.path.splitext(tile)[0] + "-rotated.jpg"
    tile_mask_name = os.path.splitext(tile)[0] + "-rotated.png"

    # Saving image and mask
    transformed['image'].save(os.path.join(output_img_dir, tile_img_name))
    transformed['mask'].save(os.path.join(output_mask_dir, tile_mask_name))
