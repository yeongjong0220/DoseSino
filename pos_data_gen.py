import torch
import os
from dataset.pos_data_loader import posDataLoader
import yaml
import numpy as np


if __name__ == "__main__":
    loader = posDataLoader() 
    
    def generate_patterns(grid_size):
        interval = 300 // grid_size
        x_list = [i * interval for i in range(grid_size + 1)]
        y_list = [i * interval for i in range(grid_size + 1)]
        sinograms = []
        for i in range(grid_size):
            for j in range(grid_size):
                image = np.zeros((300, 300)) 
                image[x_list[i]:x_list[i+1], y_list[j]:y_list[j+1]] = 1
                sinograms.append(loader.radon(image)) 
        return torch.stack(sinograms)

    sino_20 = generate_patterns(20) 
    sino_25 = generate_patterns(25) 
    
    
    full_bank = torch.cat([sino_20, sino_25], dim=0).unsqueeze(0) # [1, 1300, 352, 352]
    full_bank = (full_bank - full_bank.min()) / (full_bank.max() - full_bank.min() + 1e-8)
    

    os.makedirs("pos_data", exist_ok=True)
    np.save("pos_data/prior_sino_bank.npy", full_bank.cpu().numpy())
    print(f"Saved Hierarchical Bank: {full_bank.shape}")
