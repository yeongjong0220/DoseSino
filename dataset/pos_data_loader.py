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
        
        #full_sino = torch.tensor(sinogram).clone().detach().cpu()
        # [수정 후] sinogram이 이미 텐서이므로 torch.tensor()로 감쌀 필요 없음
        full_sino = sinogram.clone().detach().cpu()
        print(full_sino.shape)
        full_sino = (full_sino - torch.min(full_sino)) / (torch.max(full_sino) - torch.min(full_sino))
        
        return full_sino


# class posDataLoader:
#     def __init__(self, image_size=300):
#         self.image_size = image_size

#     def radon(self, images):
#         # 기존 코드 유지: Radon 변환 설정
#         detector_count = 352
#         num_view = 352
#         angles = torch.linspace(0, np.pi, num_view, device='cuda')     
#         radon_func = Radon(
#             self.image_size,
#             det_count=detector_count,
#             angles=angles,
#         )
#         images = torch.FloatTensor(images).to('cuda')
#         sinogram = radon_func.forward(images)
#         return sinogram
    
#     def get_bank_loader(self, scales=[10, 20, 30]):
#         """
#         scales: 그리드의 개수 리스트. 
#         예: 10이면 10x10(100개), 20이면 20x20(400개) 패치 생성
#         """
#         sinograms = []
        
#         for s in scales:
#             # 패치 간격 계산 (300 // 10 = 30, 300 // 20 = 15 등)
#             interval = self.image_size // s
#             x_list = [i * interval for i in range(s + 1)]
#             y_list = [i * interval for i in range(s + 1)]
            
#             print(f"Generating patterns for scale {s}x{s}...")
            
#             for i in range(s):
#                 for j in range(s):
#                     image = np.zeros((self.image_size, self.image_size))
#                     # 선택된 스케일에 따라 패치 영역 설정
#                     image[x_list[i]:x_list[i+1], y_list[j]:y_list[j+1]] = 1
#                     sinogram = self.radon(image)
#                     sinograms.append(sinogram)
        
#         # 생성된 모든 스케일의 패턴을 하나로 결합 (Batch, Channel, H, W)
#         sinogram_tensor = torch.stack(sinograms).unsqueeze(0) 
#         full_sino = sinogram_tensor.clone().detach().cpu()
        
#         # 전체 Bank에 대해 0~1 정규화 수행
#         full_sino = (full_sino - torch.min(full_sino)) / (torch.max(full_sino) - torch.min(full_sino) + 1e-8)
        
#         return full_sino