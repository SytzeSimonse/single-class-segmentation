import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def show_training_results(model, images_folder: str, output_folder: str, threshold: float = 0.2, alpha_val: float = 0.4, palette: str = 'RdYlBu'):
    # Creating paths
    images_folder_path = Path(images_folder)
    output_folder_path = Path(output_folder)

    # Creating folder for output
    if not output_folder_path.exits():
        os.mkdir(output_folder_path)
    
    # Getting list of images
    images = os.listdir(images_folder)

    for image in images:
        # Creating image path
        img_path = images_folder_path / image

        # Reading image
        img = cv2.imread(str(img_path))

        # Setting channels in right order
        img = img.transpose(2,0,1)

        # Reshaping (opposite of ravel)
        img = img.reshape(1, 3, 512, 512)

        # Displaying image
        plt.imshow(img[0,...].transpose(1,2,0))
    
        # Making prediction on image
        with torch.no_grad():
            prediction = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)

        # Getting prediction result
        result = prediction.cpu().detach().numpy()[0][0] > threshold
        
        # Overlaying prediction result
        plt.imshow(
            result, 
            alpha=alpha_val, 
            cmap=palette)

        # Saving PNG of result in inferences folder
        plt.savefig(output_folder_path / image)

        # Displaying
        plt.show()