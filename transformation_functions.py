import random
from PIL import Image, ImageEnhance
import numpy as np

# NOTE: This function changes both the mask and image.
def random_rotation(sample):
    """Rotates an image-mask pair by a random number of degrees (0-360).

    Args:
        sample (dict): Image-mask pair.

    Returns:
        dict: Transformed image-mask pair.
    """
    # Selecting random number of degrees
    degrees = random.randint(0, 360)

    # Checking if these are NumPy arrays instead of PIL images
    if type(sample['image']) == np.array():
        sample['image'] = Image.fromarray(sample['image'])
        sample['mask'] = Image.fromarray(sample['mask'])

    # Applying rotation
    sample['image'] = sample['image'].rotate(degrees)
    sample['mask'] = sample['mask'].rotate(degrees)

    return sample

# NOTE: This function changes both the mask and image.
def random_crop(sample):
    """Crops an image-mask pair (random scale: 0.6-1) to a random location.

    Args:
        sample (dict): Image-mask pair.

    Raises:
        ValueError: If image width and image height are not the same.

    Returns:
        dict: Transformed image-mask pair.
    """
    # Getting image width and image height
    image_width, image_height = sample['image'].size

    # Raising error if image width and image height differ
    if image_width != image_height:
        raise ValueError(
            f"The image height and width are not the same. This function only works with square images."
        )

    # Selecting random new crop size (smallest possible = image width * 0.6)
    crop_size = random.randint(
        int(image_width * 0.6), 
        image_width
    )

    # Setting the new frame of the image
    x = random.randint(0, image_width - crop_size) # LEFT
    y = random.randint(0, image_height - crop_size) # TOP
    x2 = x + crop_size # RIGHT
    y2 = y + crop_size # BOTTOM

    # Applying crop
    sample['image'] = sample['image'].crop(
        (x, y, x2, y2)).resize((image_width, image_height))
    sample['mask'] = sample['mask'].crop(
        (x, y, x2, y2)).resize((image_width, image_height))

    return sample

# NOTE: This function ONLY changes the image (not the mask).
def random_brightness(sample):
    """Changes the brightness of an image from an image-mask pair by a random factor (0.5-1.5).

    Args:
        sample (dict): Image-mask pair.

    Returns:
        dict: Transformed image-mask pair.
    """
    # Selecting random factor (0.5-1.5)
    factor = random.randint(50, 150) / 100

    # Changing brightness
    sample['image'] = ImageEnhance.Brightness(sample['image']).enhance(factor)

    return sample

# NOTE: This function ONLY changes the image (not the mask).
def random_contrast(sample):
    """Changes the contrast of an image from an image-mask pair by a random level (0-50).

    Args:
        sample (dict): Image-mask pair.

    Returns:
        dict: Transformed image-mask pair.
    """
    # Selecting random level
    level = random.randint(0, 50)

    # Calculating factor based on random level
    factor = (259 * (level + 255)) / (255 * (259 - level))

    # Creating function for calculating contrast by a given factor
    def contrast(c):
        return 128 + factor * (c - 128)

    # Applying contrast function to each point (i.e. pixel) in image
    sample['image'] = sample['image'].point(contrast)

    return sample

# NOTE: This function ONLY changes the image (not the mask).
def random_color(sample):
    """Changes the colour values of an image from an image-mask pair by a random factor.

    Args:
        sample (dict): Image-mask pair.

    Returns:
        sample (dict): Image-mask pair.
    """
    # Getting pixel values of image
    pixels = sample['image'].load()

    # Looping through all 'locations' in image
    for i in range(sample['image'].size[0]):  # for each column
        for j in range(sample['image'].size[1]):  # for each row
            R, G, B = pixels[i, j]
            # Assigning random new values
            pixels[i, j] = (
                R + random.randint(-100, 50),
                G + random.randint(-100, 50),
                B + random.randint(-100, 50)
            )

    return sample

def transform(sample, transformation_functions=[]):
    """Transforms a sample by applying multiple transformation functions.

    Args:
        sample (dict): Image-mask pair.
        transformation_functions (list, optional): Transformation functions.. Defaults to [].

    Returns:
        sample (dict): Image-mask pair.
    """
    # Looping through transformation functions
    for function in transformation_functions:
        # Applying transformation
        sample = function(sample)

    return sample
