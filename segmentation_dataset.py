import os
from tqdm import tqdm
from random import sample

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageCms

from torchvision.datasets.vision import VisionDataset

from transformation_functions import transform, random_rotation, random_crop, random_brightness, random_contrast

class SegmentationDataset(VisionDataset):
    """[summary]

    Args:
        root (str): Root directory path of images and masks.
        image_folder (str): Image folder name (under 'root').
        mask_folder (str): Mask folder name (under 'root').
        seed (int, optional): Seed for the train and test split (i.e. reproducible results). Defaults to None.
        fraction (float, optional): Value from 0 to 1 which specifies the validation split fraction. Defaults to None.
        subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
        transform (Optional[Callable], optional): A function/transform for the image.
        target_transform (Optional[Callable], optional): A function/transform for the mask.
        image_color_mode (str, optional): 'rgb', 'hsv', 'lab', 'ycbcr', 'rgb-hsv' or 'grayscale'. Defaults to 'rgb'.
        mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        data_augmentation: (bool): Apply data augmentation. Defaults to True.
    
    Raises:
        OSError: If image folder doesn't exist in root.
        OSError: If mask folder doesn't exist in root.
        ValueError: If subset is not either 'Train' or 'Test'
        ValueError: If image_color_mode and mask_color_mode are neither 'rgb' or 'grayscale'
    """
    def __init__(
        self, root: str, 
        image_folder: str = 'Images', 
        mask_folder: str = 'Masks',
        seed: int = None,
        fraction: float = None,
        subset: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_color_mode: str = 'rgb',
        mask_color_mode: str = 'grayscale',
        data_augmentation: bool = True) -> None:

        # Initializing the base class (i.e. VisionDataset)
        super().__init__(root)

        # Initializing class properties
        self.transform = transform
        self.target_transform = target_transform
        self.subset = subset
        self.data_augmentation = data_augmentation

        self.image_folder = image_folder
        self.mask_folder = mask_folder

        # Creating paths to image and mask folders
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder

        # Raising errors if paths to images do not exist
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        # Raising errors if selected color mode is invalid
        available_image_color_modes = ["rgb", "grayscale", "hsv", "lab", "ycbcr", "rgb-hsv", "rgb-lab", "rgb-ycbcr"]
        if image_color_mode not in available_image_color_modes:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please choose from the following modes: {', '.join(str(mode) for mode in available_image_color_modes)}."
            )
        available_mask_color_modes = ["rgb", "grayscale"]
        if mask_color_mode not in available_mask_color_modes:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please choose from the following modes: {', '.join(str(mode) for mode in available_mask_color_modes)}."
            )

        # Initiating color modes for image and mask
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))

            self.fraction = fraction

            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            # Opening image
            image = Image.open(image_file)

            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "hsv":
                image = image.convert("HSV")
            elif self.image_color_mode == "lab":
                ## Converting to LAB colour space requires a few extra steps...
                image = image.convert("RGB")

                # Convert to Lab colourspace
                srgb_p = ImageCms.createProfile("sRGB")
                lab_p  = ImageCms.createProfile("LAB")

                rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")

                image = ImageCms.applyTransform(image, rgb2lab)
            elif self.image_color_mode == "ycbcr":
                image = image.convert("YCbCr")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            elif self.image_color_mode == "rgb-hsv":
                image_rgb = image.convert("RGB")
                image_hsv = image.convert("HSV")
                image_rgb_array = np.array(image_rgb)
                image_hsv_array = np.array(image_hsv)

                # Combining ('stacking') the arrays
                image = np.dstack((image_rgb_array, image_hsv_array))
            elif self.image_color_mode == "rgb-lab":
                ## Converting to LAB colour space requires a few extra steps...
                image_rgb = image.convert("RGB")

                # Convert to Lab colourspace
                srgb_p = ImageCms.createProfile("sRGB")
                lab_p  = ImageCms.createProfile("LAB")

                rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")

                image_lab = ImageCms.applyTransform(image_rgb, rgb2lab)

                # Creating NumPy arrays
                image_rgb_array = np.array(image_rgb)
                image_lab_array = np.array(image_lab)

                # Combining ('stacking') the arrays
                image = np.dstack((image_rgb_array, image_lab_array))
            elif self.image_color_mode == "rgb-ycbcr":
                image_rgb = image.convert("RGB")
                image_ycbcr = image.convert("YCbCr")
                image_rgb_array = np.array(image_rgb)
                image_ycbcr_array = np.array(image_ycbcr)

                # Combining ('stacking') the arrays
                image = np.dstack((image_rgb_array, image_ycbcr_array))

            # Opening mask
            mask = Image.open(mask_file)

            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")

            # Creating dictionary for image-mask pair
            sample = {"image": image, "mask": mask}

            # Applying data augmentation
            if self.data_augmentation and self.subset == "Train":
                sample = transform(sample, transformation_functions = [
                  random_rotation,
                  random_brightness,
                  #random_contrast
                  random_crop                                            
                ])
            
            # Transforming
            if self.transform:
                sample["image"] = self.transform(sample["image"])

            if self.target_transform:
                sample["mask"] = self.target_transform(sample["mask"])

            return sample

    def get_empty_tiles(self) -> list:
        def is_empty_tile(tile_path: str) -> bool:
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

        # Create list for empty files
        empty_tiles = []

        # Creating path to mask folder
        mask_folder_path = Path(self.root) / self.mask_folder

        # Getting masks
        masks = os.listdir(mask_folder_path)

        # Looping through masks
        for mask in tqdm(masks, desc="Finding all empty masks..."):
            mask_path = mask_folder_path / mask
            empty_tiles.append(mask) if is_empty_tile(mask_path) else None

        return empty_tiles

    def count_empty_tiles(self) -> int:
        return len(self.get_empty_tiles())

    def count_non_empty_tiles(self) -> int:
        num_empty_tiles = self.count_empty_tiles()
        return len(self) - num_empty_tiles

    def remove_empty_tiles(self, num_of_tiles: int, balanced_classes: bool = False) -> bool:
        if balanced_classes:
            num_of_tiles = len(self) - self.count_non_empty_tiles()
        
        # Checking if the number of tiles to remove is larger than 0
        if not num_of_tiles > 0:
            raise ValueError(
                f"The number of tiles to remove should be more than 0."
            )

        # Getting empty tiles as list
        empty_tiles = self.get_empty_tiles()

        if len(empty_tiles) == 0:
            print(f"There are no empty tiles in '{self.root}'")
            return False

        # Creating paths
        image_path = Path(self.root) / self.image_folder
        mask_path = Path(self.root) / self.mask_folder

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
            for tile in tiles_to_remove:
                tile_no_extension = os.path.splitext(tile)[0]
                tile_png = tile_no_extension + '.png'
                tile_jpg = tile_no_extension + '.jpg'
                os.remove(image_path / tile_jpg)
                os.remove(mask_path / tile_png)
            return True
            
# my_dataset = SegmentationDataset("tiles", image_color_mode="rgb-hsv")
# print(my_dataset[0])

# from matplotlib import pyplot as plt
# plt.imshow(my_dataset[0]['image'][:,:,3:6], interpolation='nearest')
# plt.show()