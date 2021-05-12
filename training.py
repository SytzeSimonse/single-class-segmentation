import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

def train_model(model, criterion, dataloaders, optimizer, metrics: dict, results_path, num_epochs) -> torch.nn.Module:
    """Trains PyTorch model.

    Args:
        model (torch.nn.Module): Model to train.
        criterion (torch.nn): Loss function. 
        dataloaders (torch.utils.data.DataLoader): Dataloader.
        optimizer (torch.optim): Optimizing algorithm.
        metrics (dict): Metrics to assess model performance (from sklearn.metrics).
        results_path (str): Path to results, i.e. log of metrics and best weights.
        num_epochs (int): Number of epochs to run.

    Returns:
        torch.nn.Module: Trained model.
    """
    # Getting current time
    since = time.time()

    # Making a copy of the model's learnable parameters
    best_model_wts = copy.deepcopy(model.state_dict())

    # Creating var for best loss (initialize with extreme value)
    best_loss = 1e10

    # Using GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Putting model on device
    model.to(device)

    # Initializing the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]

    # Opening CSV file for keeping a log of the metrics
    with open(os.path.join(results_path, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        # Printing current epoch (of total epochs)
        print('Epoch {}/{}'.format(epoch, num_epochs))

        # Printing dashed line
        print('-' * 10)
        print(fieldnames)

        # Creating batchsummary as dict
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

             # Iterating over data
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                
                # Zero-ing the parameter gradients
                optimizer.zero_grad()

                # Tracking history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    for name, metric in metrics.items():
                        if name in ['f1_score', 'precision_score', 'recall_score']:
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.4))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize (only if in training phase)
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
          batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(results_path, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model (for best performance)
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # Calculating elapsed time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # Loading best model weights
    model.load_state_dict(best_model_wts)
    return model
