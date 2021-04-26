import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from random import sample

def remove_empty_tiles(root: str, image_folder: str, mask_folder: str, num_of_tiles: int):
    # Getting empty tiles as list
    empty_tiles = get_empty_tiles(root)

    if len(empty_tiles) == 0:
        print(f"There are no empty tiles in '{root}'")
        return False

    # Creating paths
    image_path = Path(root) / image_folder
    mask_path = Path(root) / mask_folder

    # Getting images and masks
    images = os.listdir(image_path)
    masks = os.listdir(mask_path)

    # Checking if number of tiles to remove is larger than number of empty tiles
    if num_of_tiles > len(empty_tiles):
        print(f"The number of tiles to remove (= {num_of_tiles}) is larger than the number of empty tiles (= {len(empty_tiles)}). All {len(empty_tiles)} tiles will be removed.")
        for tile in os.listdir(image_path):
            os.remove(image_path / tile)
        for tile in os.listdir(mask_path):
            os.remove(mask_path / tile)
    else:
        # Sampling empty tiles
        tiles_to_remove = sample(empty_tiles, num_of_tiles)
               
        # Removing tiles
        for tile in tiles_to_remove:
            tile_no_extension = os.path.splitext(tile)[0]
            tile_png = tile_no_extension + '.png'
            tile_jpg = tile_no_extension + '.jpg'
            os.remove(image_path / tile_jpg)
            os.remove(mask_path / tile_png)

def get_empty_tiles(root: str = 'tiles', mask_folder: str = 'Masks', verbose: bool = True) -> list:
    """Gets a list of all the 'empty' (=  background-only) tiles in a segmentation dataset.

    Args:
        root (str, optional): Root directory. Defaults to 'tiles'.
        mask_folder (str, optional): Name of masks folder (under root). Defaults to 'Masks'.
        verbose (bool, optional): Printing messages. Defaults to True.

    Returns:
        list: List of empty tiles.
    """
    # Create list for empty files
    empty_tiles = []

    # Creating paths
    masks_path = Path(root) / mask_folder

    # Getting masks
    masks = os.listdir(masks_path)

    # Looping through masks
    for mask in tqdm(masks, desc="Going through all masks..."):
        mask_path = masks_path / mask
        empty_tiles.append(mask) if is_empty_tile(mask_path) else None

    if verbose:
        print(f"There are {len(empty_tiles)} empty tiles in the dataset at '{root}'.")

    return empty_tiles

def is_empty_tile(tile_path: str) -> bool:
    """Checks whether a tile is empty (= only background).

    Args:
        tile_path (str): Tile filepath.

    Returns:
        bool: True if tile is empty, false if tile is non-empty.
    """
    # Creating path of tile file
    tile_fpath = Path(tile_path)

    # Checking if file exists
    if not tile_fpath.exists():
        raise FileNotFoundError(
            f"'{tile_path}' does not exist."
        )

    # Checking if image extension
    allowed_extensions = ['.png', '.jpg', '.jpeg']
    tile_extension = os.path.splitext(tile_fpath)[1]

    if not tile_extension in allowed_extensions:
        raise ValueError(
            f"'{tile_extension}' files cannot be used with this function."
        )

    # Opening image
    img = Image.open(tile_path)

    # Reading into NumPy
    img_array = np.asarray(img)

    # Check if all values are zero
    if np.all((img_array == 0)):
        return True
    return False

#remove_empty_tiles(root='test_remove', image_folder='Images', mask_folder='Masks', num_of_tiles=1)

print(len(get_empty_tiles()))

#is_empty_tile('/home/sytze/Code/DeepLabV3Plus_implementation-single-class/test_remove/Masks/tile_pinet1-2019_2048_5632.png')
#is_empty_tile('/home/sytze/Code/DeepLabV3Plus_implementation-single-class/test_remove/Masks/tile_pinet1-2019_2048_5120.png')