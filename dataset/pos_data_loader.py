import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
# import odl
from torch_radon import Radon
from skimage.transform import radon, iradon, resize

class posDataset(Dataset):
    def __init__(self):
        """
        Args:
            dataset_path (str): Path to the dataset directory.
            ids (list): List of folder names to load data from.
        """
                
    def radon(self, images):
        image_size = images.shape[-1]
        detector_count = 352
        num_view = 352
        
        angles = torch.linspace(0, np.pi, num_view, device='cuda')     
        radon = Radon(
            image_size,
            det_count=detector_count,
            angles=angles,
        )
        images = torch.FloatTensor(images).to('cuda')
        sinogram = radon.forward(images)
        
        return sinogram

    def __len__(self):
        return len(np.array([1]))

    def __getitem__(self):

        # Normalize the image data
        # Fill each channel with a unique 16x16 block set to 1
        x_list = [i for i in range(0, 401, 20)]
        y_list = [i for i in range(0, 401, 20)]
        
        sino = np.zeros((400, 360, 360))
        sinograms = []
        for i in range(20):
            for j in range(20):
                image = np.zeros(400, 400)
                image[x_list[i]:x_list[i+1], y_list[j]:y_list[j+1]] = 1
                sinogram = self.radon(image)
                sinograms.append(sinogram)
        
        sinogram = torch.stack(sino)
        print(sinogram.shape)
        
        full_sino = torch.tensor(sinogram)
        full_sino = (full_sino - torch.min(full_sino)) / (torch.max(full_sino) - torch.min(full_sino))
        
        
        return full_sino


class posDataLoader:
    def __init__(self, batch_size=1, shuffle=True, num_workers=0, train_val="train"):
        """
        Args:
            dataset (Dataset): Custom dataset class instance (AAPMDataset).
            batch_size (int): Number of samples per batch.
            shuffle (bool): If True, shuffle the data.
            num_workers (int): Number of subprocesses to use for data loading.
        """


    def radon(self, images):
        image_size = images.shape[-1]
        detector_count = 352
        num_view = 352
        
        angles = torch.linspace(0, np.pi, num_view, device='cuda')     
        radon = Radon(
            image_size,
            det_count=detector_count,
            angles=angles,
        )
        images = torch.FloatTensor(images).to('cuda')
        sinogram = radon.forward(images)
        
        return sinogram
    
    def get_loader(self):
        # Normalize the image data
        x_list = [i for i in range(0, 301, 15)]
        y_list = [i for i in range(0, 301, 15)]
        
        print(len(x_list))
        print(len(y_list))
        
        sinograms = []
        for i in range(20):
            for j in range(20):
                image = np.zeros((300, 300))
                image[x_list[i]:x_list[i+1], y_list[j]:y_list[j+1]] = 1
                sinogram = self.radon(image)
                sinograms.append(sinogram)
        
        sinogram = torch.stack(sinograms).unsqueeze(0)
        full_sino = sinogram.clone().detach().cpu()
        print(full_sino.shape)
        full_sino = (full_sino - torch.min(full_sino)) / (torch.max(full_sino) - torch.min(full_sino))
        
        return full_sino
