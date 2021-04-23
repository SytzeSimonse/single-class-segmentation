import os
import tarfile

import urllib.request
from tqdm import tqdm

from pathlib import Path

def download_tiles(
    tiles_folder_name="tiles",
    data_folder_name="data",
    data_url: str = None,
    show_progress: bool = True
) -> None:

    # Creating paths to image and mask folders
    tiles_folder_path = Path(tiles_folder_name)
    data_folder_path = Path(data_folder_name)
    data_path = data_folder_path / "data.tar.gz"

    # Creating folders if not exist
    if not tiles_folder_path.exists():
        os.mkdir(tiles_folder_path)
    if not data_folder_path.exists():
        os.mkdir(data_folder_path)

    # Downloading data
    if not data_path.exists():
        download_url(
            url=data_url,
            output_path=data_path
        )

    # Opening file
    data_tar = tarfile.open(data_path)

    # Extracting file
    data_tar.extractall(tiles_folder_path)

    # Closing file
    data_tar.close()

    # Getting paths to images and masks resp.
    images_path = tiles_folder_path / 'Images'
    masks_path = tiles_folder_path / 'Masks'

    # Counting total number of images and masks
    num_of_images = len(os.listdir(images_path))
    num_of_masks = len(os.listdir(masks_path))

    # Checking if image-mask pairs are 'complete'
    if num_of_images != num_of_masks:
        raise ValueError(
            f"There are {num_of_images} images, but {num_of_masks}."
        )

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to
            )
