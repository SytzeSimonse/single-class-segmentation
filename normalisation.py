import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def calculate_RGB_images_statistic(input_dir, operation=np.mean) -> float:
    """Calculates statistics of (a set of) RGB (i.e. 3-channel) images with a given NumPy function.

    Args:
        input_dir (str): Directory with images.
        operation (NumPy statistic function, optional): Function to apply. Defaults to np.mean.

    Returns:
        float: Tuple with calculated statistic for each channel.
    """
    # Listing images
    images = os.listdir(input_dir)
    
    # Checking if directory is not empty
    if len(images) == 0:
        raise ValueError(
            f"'{input_dir}' is empty, so no statistics could be calculated."
        ) 

    # Initializing vars for RGB-channels
    R = 0
    G = 0
    B = 0

    # Looping through images
    for image in tqdm(images, desc="Calculating image statistics..."):
        # Getting filepath of image
        img_fp = os.path.join(input_dir, image)

        # Opening image
        img = Image.open(img_fp)

        # Reading image into NumPy array
        np_img = np.asarray(img)
        
        # Applying operation to each channel
        R = R + operation(np_img[:,:,0])
        G = G + operation(np_img[:,:,1])
        B = B + operation(np_img[:,:,2])

    # Dividing statistics by number of images and normalizing (division by 255)
    R = R / len(images) / 255
    G = G / len(images) / 255
    B = B / len(images) / 255

    # Putting RGB-channels into tuple
    channels = [R, G, B]

    return channels
