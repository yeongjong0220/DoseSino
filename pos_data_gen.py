import torch
import os
from dataset.pos_data_loader import posDataLoader
import yaml
import numpy as np


# if __name__ == "__main__":

    
#     full_data = posDataLoader(batch_size=1, shuffle=False, num_workers=0, train_val="train").get_loader()
    
#     save_dir = "pos_data"
#     os.makedirs(save_dir, exist_ok=True) 

#     # 루프 돌지 말고 바로 저장!
#     np.save(os.path.join(save_dir, "prior_sino.npy"), full_data.numpy())
#     print(f"Saved prior_sino.npy with shape: {full_data.shape}")

#     # for i, img in enumerate(full_data):
#     #     img = np.array(img)
#     #     print(img.shape)
#     #     np.save(os.path.join(save_dir, f"prior_sino.npy"), img)

# [pos_data_gen.py] 전체 수정 로직
if __name__ == "__main__":
    loader = posDataLoader() # pos_data_loader.py 내 클래스 호출
    
    def generate_patterns(grid_size):
        interval = 300 // grid_size
        x_list = [i * interval for i in range(grid_size + 1)]
        y_list = [i * interval for i in range(grid_size + 1)]
        sinograms = []
        for i in range(grid_size):
            for j in range(grid_size):
                image = np.zeros((300, 300)) # [수정] 튜플 사용
                image[x_list[i]:x_list[i+1], y_list[j]:y_list[j+1]] = 1
                sinograms.append(loader.radon(image)) # GPU 연산 수행
        return torch.stack(sinograms)

    # 1. 스케일별 생성
    sino_20 = generate_patterns(20) # 400개
    sino_25 = generate_patterns(25) # 625개 원래는 900개
    
    # 2. 통합 및 정규화 (총 1,300개)
    full_bank = torch.cat([sino_20, sino_25], dim=0).unsqueeze(0) # [1, 1300, 352, 352]
    full_bank = (full_bank - full_bank.min()) / (full_bank.max() - full_bank.min() + 1e-8)
    
    # 3. 저장
    os.makedirs("pos_data", exist_ok=True)
    np.save("pos_data/prior_sino_bank.npy", full_bank.cpu().numpy())
    print(f"Saved Hierarchical Bank: {full_bank.shape}")