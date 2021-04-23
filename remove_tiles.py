from random import sample
from pathlib import Path
import os

def randomly_remove_tiles(root, image_folder, mask_folder, num_of_tiles, verbose=True):
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
        print(f"The number of tiles has been reduced from {image_count} to {image_count-num_of_tiles}."