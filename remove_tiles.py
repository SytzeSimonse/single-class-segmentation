from random import sample
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image

def randomly_remove_tiles(root: str, image_folder: str, mask_folder: str, num_of_tiles: int, verbose=True):
    # Creating paths
    image_folder_path = Path(root) / image_folder
    mask_folder_path = Path(root) / mask_folder

    # Counting images and masks resp.
    image_count = len(os.listdir(image_folder_path))
    mask_count = len(os.listdir(mask_folder_path))

    if not image_count == mask_count:
        raise ValueError(
            f"The number of images (= {image_count}) is not equal to the number of masks (= {mask_count})."
        )

    if num_of_tiles == image_count:
        raise ValueError(
            f"The number of tiles to remove is equal to the total number of tiles. This action would delete all tiles."
        )

    if num_of_tiles > image_count:
        raise ValueError(
            f"The number of tiles cannot be larger than the total number of tiles."
        )

    if num_of_tiles == 0:
        raise ValueError(
            f"The number of tiles to remove cannot be equal to 0."
        )

    # Sampling indeces
    indices = sample(range(0, image_count), num_of_tiles)

    # Removing tiles
    for idx in indices:
        os.remove(image_folder_path / sorted(os.listdir(image_folder_path))[idx])
        os.remove(mask_folder_path / sorted(os.listdir(mask_folder_path))[idx])

    if verbose:
        print(f"The number of tiles has been reduced from {image_count} to {image_count-num_of_tiles}.")

def randomly_remove_empty_tiles(root: str, image_folder: str, mask_folder: str, num_of_tiles: int) -> bool:
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

        return True
    else:
        # Sampling empty tiles
        tiles_to_remove = sample(empty_tiles, num_of_tiles)
               
        # Removing tiles
        for tile in tqdm(tiles_to_remove, desc="Removing tiles..."):
            tile_no_extension = os.path.splitext(tile)[0]
            tile_png = tile_no_extension + '.png'
            tile_jpg = tile_no_extension + '.jpg'
            os.remove(image_path / tile_jpg)
            os.remove(mask_path / tile_png)

        return True

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