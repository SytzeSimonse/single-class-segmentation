from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from segmentation_dataset import SegmentationDataset

def create_dataloader(
    root: str = 'Tiles', 
    images_folder_name: str = 'Images',
    image_color_mode: str = "rgb",
    masks_folder_name: str = 'Masks',
    seed: int = 100,
    fraction: float = 0.2,
    batch_size: int = 8,
    workers: int = 2 
    ) -> dict:
    """Creates dataloaders for training and test phase.

    Args:
        root (str, optional): Root directory. Defaults to 'Tiles'.
        images_folder_name (str, optional): Images folder (under root). Defaults to 'Images'.
        masks_folder_name (str, optional): Masks folder (under root). Defaults to 'Masks'.
        seed (int, optional): Random seed. Defaults to 100.
        fraction (float, optional): Split for training/testing. Defaults to 0.2.
        batch_size (int, optional): Batch size. Defaults to 8.
        workers (int, optional): Number of processes running simultaneously. Defaults to 2 (recommendation by Google Colab).

    Returns:
        dict: Dataloaders for training and test phase.
    """
    # Creating the dataloader
    image_datasets = {
            phase: SegmentationDataset(
                root=root,
                image_folder=images_folder_name,
                mask_folder=masks_folder_name,
                seed=seed,
                fraction=fraction,
                subset=phase,
                # Converting to tensors by default
                transform=transforms.ToTensor(), 
                target_transform=transforms.ToTensor(),
                image_color_mode=image_color_mode,
                mask_color_mode='grayscale'
                )

            for phase in ['Train', 'Test']
        }

    dataloaders = {
            phase: DataLoader(
                image_datasets[phase],
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers
                )
            
            for phase in ['Train', 'Test']
        }

    return dataloaders