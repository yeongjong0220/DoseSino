import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torch_radon import Radon
from skimage.transform import radon, iradon, resize
from PIL import Image
import random
from scipy.ndimage import rotate

import nibabel as nib


class SinoDataset_N_limit_dose(Dataset):
    def __init__(self, data_root, json_path, mi_root_dir=None, mode='train'):
        """
        mi_root_dir: 'mi_results' 폴더 경로
        """
        self.data_root = data_root
        self.mode = mode
        
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
            
        self.dose_value_map = {
            "Full_dose": 100.0,
            "1-2_dose": 50.0,
            "1-4_dose": 25.0,
            "1-10_dose": 10.0,
            "1-20_dose": 5.0,
            "1-50_dose": 2.0,
            "1-100_dose": 1.0
        }
        
        self.mi_root_dir = mi_root_dir
        self.mi_cache = {} 
        self.INPUT_LEN = 36 
        
        if self.mi_root_dir and os.path.exists(self.mi_root_dir):
            self._preload_mi_data()
        else:
            print("⚠️ Warning: MI root directory not found. MI vectors will be zeros.")

    def _preload_mi_data(self):
        print(f"Loading MI data from {self.mi_root_dir}...")
        subjects = [d for d in os.listdir(self.mi_root_dir) if d.startswith("Subject")]
        
        for subj in subjects:
            json_file = os.path.join(self.mi_root_dir, subj, 'mi_distributions_raw.json')
            if not os.path.exists(json_file): continue
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                for pid, doses in data.items():
                    if pid not in self.mi_cache: self.mi_cache[pid] = {}
                        
                    for dose_key_raw, items in doses.items():
                        dose_key_norm = dose_key_raw.replace(' ', '_')
                        mi_vectors = []
                        for item in items:
                            mi = item.get('mi', [])
                            proc_mi = self._process_mi_vector(mi)
                            mi_vectors.append(proc_mi)
                        self.mi_cache[pid][dose_key_norm] = mi_vectors
            except Exception as e:
                print(f"Error loading MI for {subj}: {e}")
        print("✅ MI Data preloaded.")

    def _process_mi_vector(self, vector):
        target_length = self.INPUT_LEN
        if not vector: return np.zeros(target_length, dtype=np.float32)
        vector = np.array(vector)
        vector = vector[np.isfinite(vector)]
        if len(vector) < target_length:
            vector = np.pad(vector, (0, target_length - len(vector)), 'constant')
        elif len(vector) > target_length:
            vector = vector[:target_length]
        return np.log1p(vector).astype(np.float32)

    def __getitem__(self, index):
        item = self.data_list[index]
        
        input_path = os.path.join(self.data_root, item['input'])
        target_path = os.path.join(self.data_root, item['target'])
        
        input_data = np.load(input_path)
        target_data = np.load(target_path)
        
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        
        if input_tensor.dim() == 2: input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 2: target_tensor = target_tensor.unsqueeze(0)

        file_name = item['file_name']
        
        if 'seg' in item:
             seg_path = os.path.join(self.data_root, item['seg'])
             seg_label = np.load(seg_path)
             seg_label = torch.from_numpy(seg_label).long()
        else:
            seg_label = torch.tensor(0).long() 

        # --- Dose & MI Extraction ---
        parts = input_path.split(os.sep)
        subj_id = "Unknown"
        dose_key = "Full_dose"
        
        for part in parts:
            if part.startswith("Subject"): subj_id = part
            if "dose" in part: dose_key = part
        
        dose_val = self.dose_value_map.get(dose_key, 100.0)
        mi_vec = np.zeros(self.INPUT_LEN, dtype=np.float32)
        
        if self.mi_cache and subj_id in self.mi_cache:
            if dose_key in self.mi_cache[subj_id]:
                mi_list = self.mi_cache[subj_id][dose_key]
                
                # [수정된 부분] 파일명에서 숫자 추출 (Regex)
                # 예: "slice_005.npy" -> ['005'] -> 5
                # 예: "123.npy" -> ['123'] -> 123
                numbers = re.findall(r'\d+', file_name)
                
                if numbers:
                    # 파일명에 숫자가 여러 개면 가장 마지막 숫자(보통 슬라이스 번호) 사용
                    slice_idx = int(numbers[-1])
                    
                    # [주의] 데이터셋 인덱싱 보정
                    # 만약 파일명이 1부터 시작하는데 리스트는 0부터 시작하면: slice_idx - 1
                    # 여기서는 0부터 시작한다고 가정하고 그대로 씀 (필요시 -1 추가하세요)
                    final_idx = slice_idx 
                    
                    if 0 <= final_idx < len(mi_list):
                        mi_vec = mi_list[final_idx]

        return {
            'input': input_tensor,
            'target': target_tensor,
            'seg': seg_label,
            'file_name': file_name,
            'mi_vec': torch.from_numpy(mi_vec).float(),
            'dose': torch.tensor(dose_val).float()
        }

    def __len__(self):
        return len(self.data_list)




"""기존 low dose를 입력 받는 코드"""
class AAPMDataset(Dataset):
    def __init__(self, dataset_path, ids, bank_path, augment=True):
        """
        Args:
            dataset_path (str): Path to the dataset directory.
            ids (list): List of folder names to load data from.
        """
        self.dataset_path = dataset_path
        self.augment = augment
        
        self.data_file_paths = []
        for folder_id in ids:
            folder_path = os.path.join(self.dataset_path, folder_id)
            self.data_file_paths.extend(glob(os.path.join(folder_path, '*.npy')))
        # for folder_id in ids:
        #     folder_path = os.path.join(self.dataset_path, folder_id)
        #     self.data_file_paths.extend(glob(os.path.join(folder_path, '*.png')))
        # self.pos_data = np.load(bank_path)
        
    def apply_augmentation(self, image, label):
        """
        Apply augmentation (rotation, flip, etc.) to the image.
        Args:
            image (numpy.ndarray): Input image with shape (1, h, w).
        Returns:
            numpy.ndarray: Augmented image.
        """
        # Random rotation
        if random.random() < 0.5:
            # angle = random.choice([90, 180, 270])
            angle = random.uniform(-45, 45)  # Rotate within ±30 degrees
            image = rotate(image.squeeze(0), angle=angle, order=0, reshape=False)
            label = rotate(label.squeeze(0), angle=angle, order=0, reshape=False)
            # image = np.rot90(image.squeeze(0), k=angle // 90)#.copy()  # Ensure no negative stride by copying the array
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
        # Random horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, axis=2)#.copy()  # Ensure no negative stride by copying the array
            label = np.flip(label, axis=2)
            
        return image, label
    
    def radon(self, images, num_view=352):
        
        image_size = images.shape[-1]
        detector_count = 352

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
        return len(self.data_file_paths)

    # def __getitem__(self, idx):
    #     # Load the .npy file
    #     sino_path = self.data_file_paths[idx]
    #     dose = [int(sino_path.split("_sino_")[1][:-4])]
    #     image_path = sino_path.replace("input", "original")
    #     label_path = sino_path.replace("input", "labels")
    #     label_path = label_path.split("_0001")[0] + "_label.npy"
    #     prefix = image_path.split("_sino_")[0]
    #     image_path = prefix + "_label.npy"
    #     full_sino_path = prefix + "_sino_full.npy"
    #     # label_path = img_path.replace("image", "label")
    #     # image = np.load(img_path)
    #     file_name = os.path.basename(sino_path)
    #     image = np.array(np.load(image_path))
    #     full_sino = np.array(np.load(full_sino_path))
    #     sino = np.array(np.load(sino_path))  # Convert to grayscale
    #     label = np.array(np.load(label_path))
    #     # label = np.array(np.load(label_path))
    #     # image = np.expand_dims(image, axis=0)
    #     # label = np.expand_dims(label, axis=0)
    #     # label = np.expand_dims(label, axis=0)
        
    #     # Normalize the image data
    #     # image = resize(image, (1, 512, 512), anti_aliasing=True)
        
    #     # Apply data augmentation if enabled
    #     if self.augment:
    #         image, label = self.apply_augmentation(image, label)
    #     if np.max(image) != np.min(image):
    #         image = (image - np.min(image)) / (np.max(image) - np.min(image))
    #     # label = label / 255.
    #     # label = np.where(label > 0, 1, 0)
    #     # print(image.shape)
        
    #     # sino_label = self.radon_fanbeam(label)        
    #     # Convert to tensor
        
    #     dose = torch.tensor(dose, dtype=torch.float32) / 100
    #     image_tensor = torch.tensor(image, dtype=torch.float32)
    #     full_sino = torch.tensor(full_sino)
    #     input_sino = torch.tensor(sino)
    #     label = torch.tensor(label)

    #     # Full Dose Max가 약 80까지 나오므로 100으로 여유있게 설정
    #     GLOBAL_MAX = 100.0
    #     GLOBAL_MIN = 0.0
    #     # Low Dose와 Full Dose의 비율 차이가 약 10~15배이므로 대충 13으로 보정
    #     SCALE_FACTOR = 13.0
    #     # 3. 정규화 수행
    #     # 입력(Low Dose)을 13배 튀겨서 Full Dose와 비슷한 체급으로 만든 뒤 100으로 나눔
    #     # 결과: 둘 다 0.0 ~ 1.0 사이의 안전한 범위에 안착함
    #     input_sino_norm = (input_sino * SCALE_FACTOR - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
    #     full_sino_norm = (full_sino - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
    #     # 이제 개별 max가 아니라 고정된 GLOBAL_MAX를 전달
    #     max_value = torch.tensor([GLOBAL_MAX], dtype=torch.float32)
    #     min_value = torch.tensor([GLOBAL_MIN], dtype=torch.float32)
    #     #print(max_value.shape)
    #     #max_value = torch.max(input_sino)
    #     #min_value = torch.min(input_sino)
    #     #full_max = torch.max(full_sino)
    #     #full_min = torch.min(full_sino)
    #     #print("full_max, full_min", full_max, full_min) # min=0 max=40~70 다양
    #     #full_sino = (full_sino - full_max) / (full_max - full_min + 1e-8)
    #     #input_sino = (input_sino - full_min) / (full_max - full_min + 1e-8)
        
    #     #max_value = max_value[...,None,None]
    #     #min_value = min_value[...,None,None]
    #     max_value = max_value[...,None]
    #     min_value = min_value[...,None]
    #     #print("max_value, min_value", max_value, min_value) # min=0 max=1~7 다양
    #     # sino_label = torch.tensor(sino_label)
    #     sino_label = torch.tensor(0)
        
    #     return image_tensor, full_sino_norm, input_sino_norm, max_value, min_value, sino_label, label, file_name, dose
    def __getitem__(self, idx):
            # 1. 파일 경로 파싱 및 생성
            sino_path = self.data_file_paths[idx]
            
            # 공통 Prefix 추출 (예: .../input/filename_0001_527)
            prefix_input = sino_path.split("_sino_")[0]
            
            # 각 데이터의 경로 생성
            # Image (GT): input -> original, suffix -> _image.npy
            image_path = prefix_input.replace("input", "original") + "_image.npy"
            
            # Full Sino (GT): input -> original, suffix -> _sino_full.npy
            full_sino_path = prefix_input.replace("input", "original") + "_sino_full.npy"
            
            # Label (Seg): input -> labels, suffix -> _label_slice.npy
            label_path = prefix_input.replace("input", "labels") + "_label_slice.npy"

            file_name = os.path.basename(sino_path)
            # ------------------------------------------------------------------
            # [수정] Dose 변수 초기화 (에러 방지)
            # ------------------------------------------------------------------
            dose_val = 100.0 # 기본값 설정 (변수 선언 보장)
            
            try:
                if "_sino_10." in file_name or "_sino_10_" in file_name:
                    dose_val = 10.0
                elif "_sino_50" in file_name:
                    dose_val = 50.0
                elif "_sino_1" in file_name:
                    dose_val = 1.0
                elif "_sino_25" in file_name:
                    dose_val = 25.0
                elif "_sino_5" in file_name:
                    dose_val = 5.0
                elif "full" in file_name.lower():
                    dose_val = 100.0
            except:
                dose_val = 100.0 # 파싱 실패 시 기본값`

            # 2. 데이터 로드
            try:
                image = np.array(np.load(image_path))       # 300x300 (이미 잘려있음)
                full_sino = np.array(np.load(full_sino_path))
                sino = np.array(np.load(sino_path))
                label = np.array(np.load(label_path))       # 400x400 (안 잘려있음 -> 문제 원인)
                
                # --- [핵심 수정: Label 크기 맞추기] ---
                # data_processing.py에서 [50:-50, 50:-50]으로 잘랐으므로 똑같이 잘라줍니다.
                if label.shape[0] == 400 and label.shape[1] == 400:
                    label = label[50:-50, 50:-50]
                # ------------------------------------
                
            except FileNotFoundError as e:
                print(f"\n[Error] 파일을 찾을 수 없습니다: {e.filename}")
                print(f"  - Source: {sino_path}")
                raise e

            # 3. 데이터 증강 및 이미지 정규화
            if self.augment:
                image, label = self.apply_augmentation(image, label)
            
            if np.max(image) != np.min(image):
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            # 4. Tensor 변환
            # [수정] 위에서 초기화한 dose_val을 사용하여 텐서 생성
            dose_tensor = torch.tensor([dose_val], dtype=torch.float32)
            image_tensor = torch.tensor(image, dtype=torch.float32)
            full_sino = torch.tensor(full_sino, dtype=torch.float32)
            input_sino = torch.tensor(sino, dtype=torch.float32)
            label = torch.tensor(label)
            # print("input max, min",input_sino.max(),input_sino.min())  # min=0, max=0.5~5
            # print("full max, min",full_sino.max(),full_sino.min())  # min=0, max=5.8~70
            # print("label max min", label.max(),label.min()) # min=0 max=1
            # 1. Input Normalization Factor (입력용)
            # 로그 분석 결과: 저선량 Max는 대부분 1~4 사이, 튀어도 6 미만임.
            # 따라서 10.0으로 나누면 0.1~0.6 사이로 아주 예쁘게 들어옴.
            if dose_val < 99.0:
                INPUT_NORM_FACTOR = 10.0  
            else:
                # 만약 입력이 Full Dose라면 타겟과 똑같이 처리
                INPUT_NORM_FACTOR = 100.0 

            # 2. Target Normalization Factor (정답용)
            # 로그 분석 결과: Full Dose Max는 10~40 사이 (가끔 70).
            # 100.0으로 나누면 0.1~0.7 사이로 안전하게 들어옴.
            TARGET_NORM_FACTOR = 100.0

            # 3. 정규화 수행 (Clamp로 0~1 범위 안전장치)
            # 입력: 4.15 -> 0.415
            # 타겟: 39.0 -> 0.390
            # -> 네트워크는 비슷한 크기의 숫자를 다루게 되어 학습이 매우 잘 됨.
            input_sino_norm = torch.clamp(input_sino / INPUT_NORM_FACTOR, 0.0, 1.0)
            full_sino_norm = torch.clamp(full_sino / TARGET_NORM_FACTOR, 0.0, 1.0)
            
            # 4. 모델에 전달할 복원 정보 (가장 중요!)
            # 모델은 0~1 사이 값을 예측함. 이를 물리적 수치(Max 40)로 되돌리려면
            # 반드시 'Target을 나눴던 값(100.0)'을 곱해줘야 함.
            max_value = torch.tensor([TARGET_NORM_FACTOR], dtype=torch.float32)
            min_value = torch.tensor([0.0], dtype=torch.float32)

            # # 5. Global Max Normalization 설정
            # GLOBAL_MAX = 100.0
            # GLOBAL_MIN = 0.0
            # SCALE_FACTOR = 13.0

            # # 6. 정규화 수행
            # # 입력(Low Dose) 스케일링 및 정규화 (변수명 input_sino 사용)
            # input_sino_norm = (input_sino * SCALE_FACTOR - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
            
            # # 타겟(Full Dose) 정규화 (변수명 full_sino 사용)
            # full_sino_norm = (full_sino - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
            
            # # 7. 모델에 전달할 복원 정보 (고정값)
            # max_value = torch.tensor([GLOBAL_MAX], dtype=torch.float32)
            # min_value = torch.tensor([GLOBAL_MIN], dtype=torch.float32)
            
            
            # # (3) 모델에 전달할 Max/Min 값 (Denormalization용)
            # # 이제 Global 값이 아니라, 해당 이미지의 실제 Max/Min을 전달해야 복원이 정확합니다.
            # max_value = torch.tensor([in_max], dtype=torch.float32) # GLOBAL_MAX (X) -> in_max (O)
            # min_value = torch.tensor([in_min], dtype=torch.float32) # GLOBAL_MIN (X) -> in_min (O)

            # 차원 맞춤 [Batch, 1, 1] (DataLoader가 Batch 차원 추가함)
            max_value = max_value[..., None]
            min_value = min_value[..., None]

            sino_label = torch.tensor(0)
            
            return image_tensor, full_sino_norm, input_sino_norm, max_value, min_value, sino_label, label, file_name, dose_tensor


"""full dose를 입력 받는 코드"""
# class AAPMDataset(Dataset):
#     def __init__(self, dataset_path, ids, bank_path, augment=True):
#         """
#         Args:
#             dataset_path (str): Path to the dataset directory.
#             ids (list): List of folder names to load data from.
#         """
#         self.dataset_path = dataset_path
#         self.augment = augment
        
#         self.data_file_paths = []
#         for folder_id in ids:
#             folder_path = os.path.join(self.dataset_path, folder_id)
            
#             # 1. 모든 .npy 파일을 불러옵니다.
#             all_npy_files = glob(os.path.join(folder_path, '*.npy'))
            
#             # 2. [핵심 수정] 불러온 파일 중 '_sino_'를 포함하는 유효한 시노그램 파일만 필터링합니다.
#             #    이로써 '..._label.npy' 파일이 목록에 포함되어 충돌하는 것을 방지합니다.
#             valid_sino_files = [f for f in all_npy_files if '_sino_' in f]
            
#             # 3. 유효한 파일 목록을 최종 리스트에 추가합니다.
#             self.data_file_paths.extend(valid_sino_files)
            
#         if not self.data_file_paths:
#             raise RuntimeError("No valid sinogram input files found with '_sino_' identifier.")

#         # self.pos_data = np.load(bank_path)
        
#     def apply_augmentation(self, image, label):
#         """
#         Apply augmentation (rotation, flip, etc.) to the image.
#         Args:
#             image (numpy.ndarray): Input image with shape (1, h, w).
#         Returns:
#             numpy.ndarray: Augmented image.
#         """
#         # Random rotation
#         if random.random() < 0.5:
#             angle = random.uniform(-45, 45)  
#             image = rotate(image.squeeze(0), angle=angle, order=0, reshape=False)
#             label = rotate(label.squeeze(0), angle=angle, order=0, reshape=False)
#             image = np.expand_dims(image, axis=0)
#             label = np.expand_dims(label, axis=0)
#         # Random horizontal flip
#         if random.random() < 0.5:
#             image = np.flip(image, axis=2)
#             label = np.flip(label, axis=2)
            
#         return image, label
    
#     def radon(self, images, num_view=512):
        
#         image_size = images.shape[-1]
#         detector_count = 512

#         angles = torch.linspace(0, np.pi, num_view, device='cuda')
#         radon = Radon(
#             image_size,
#             det_count=detector_count,
#             angles=angles,
#         )
        
#         images = torch.FloatTensor(images).to('cuda')
#         sinogram = radon.forward(images)

#         return sinogram
        
#     def __len__(self):
#         return len(self.data_file_paths)

#     def __getitem__(self, idx):
#         sino_path = self.data_file_paths[idx]
        
#         # --- [안전 검사 및 경로 재구성 START] ---
        
#         # 1. Dose 추출 및 유효성 검사
#         try:
#             dose_parts = sino_path.split("_sino_")
#             if len(dose_parts) <= 1:
#                 # _sino_가 없으면 (라벨 파일이나 원본 이미지 파일일 경우) 처리를 중단하고 오류 발생
#                 raise ValueError("File path missing '_sino_' delimiter.")
                
#             # 선량 추출
#             dose_str_with_ext = dose_parts[1]
#             dose_str = os.path.splitext(dose_str_with_ext)[0]
            
#             if dose_str.isdigit():
#                 dose_val = [int(dose_str)]
#             elif 'full' in dose_str.lower():
#                 dose_val = [100]
#             else:
#                 dose_val = [100]
#         except Exception as e:
#              # DataLoader가 이 파일을 건너뛰도록 유도하는 RuntimeError 발생
#              raise RuntimeError(f"Skipping corrupt file {sino_path}: {e}") from e

#         # 1. 고유 접두사 추출 및 디렉토리 치환
#         prefix = sino_path.split("_sino_")[0]
#         original_prefix = prefix.replace("input", "original") 
        
#         # 2. 필수 파일 경로 구성 (존재하는 파일만 정의)
#         label_path = original_prefix + "_label.npy"      # GT Image & Seg Mask
#         full_sino_path = original_prefix + "_sino_full.npy" # Clean Target Sinogram
        
#         # 3. 데이터 로드 (FileNotFoundError가 발생했던 지점)
#         file_name = os.path.basename(sino_path)
        
#         try:
#             # [수정 1] label_array를 두 번 로드 (GT Image와 Seg Mask 역할을 겸함)
#             # image_array (PSNR용) <- label_path 로드
#             image_array = np.array(np.load(label_path)) 
#             label_array = np.array(np.load(label_path)) 
#             full_sino_array = np.array(np.load(full_sino_path))
#             sino_low_dose_array = np.array(np.load(sino_path)) 
#         except FileNotFoundError as e:
#             # 경로 구성 오류가 발생하면 건너뛰도록 유도
#             raise RuntimeError(f"Failed to find companion file: {e.filename}") from e


#         # 4. 증강 및 이미지 정규화
#         if self.augment:
#             image_array, label_array = self.apply_augmentation(image_array, label_array)
#         if np.max(image_array) != np.min(image_array):
#             image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))


#         # 5. Tensor 변환 및 정규화
        
#         # Full Sino의 Min/Max로 정규화 (GT Target)
#         full_max = torch.max(torch.tensor(full_sino_array))
#         full_min = torch.min(torch.tensor(full_sino_array))
        
#         full_sino_norm = (torch.tensor(full_sino_array) - full_min) / (full_max - full_min + 1e-8)
        
#         # 모델 입력 (Input Sino)으로 Full Dose를 사용하도록 설정
#         input_sino_norm = (torch.tensor(full_sino_array) - full_min) / (full_max - full_min + 1e-8) 

#         # TransSino Denormalization에 필요한 Low-Dose Input의 Max/Min 사용 (안정성을 위해)
#         max_value_tensor = torch.max(torch.tensor(sino_low_dose_array))
#         min_value_tensor = torch.min(torch.tensor(sino_low_dose_array))
        
#         # 6. Final Return
#         dose = torch.tensor(dose_val, dtype=torch.float32) / 100
#         image_tensor = torch.tensor(image_array, dtype=torch.float32)
        
#         # Max/Min 차원 조정
#         max_value = max_value_tensor[...,None,None]
#         min_value = min_value_tensor[...,None,None]
#         sino_label = torch.tensor(0)
        
#         return image_tensor, full_sino_norm, input_sino_norm, max_value, min_value, sino_label, torch.tensor(label_array), file_name, dose



class NiftiSliceDataset(Dataset):
    def __init__(self, base_dir: str):
        self.paths = []
        # .nii와 .nii.gz 파일을 재귀적으로 탐색
        for ext in ("*.nii", "*.nii.gz"):
            self.paths.extend(glob(os.path.join(base_dir, "**", ext), recursive=True))
        
        if not self.paths:
            raise RuntimeError(f"지정된 경로에서 NIfTI 파일을 찾을 수 없습니다: {base_dir}")

        self.slice_map = []
        self.cumulative_slices = [0]
        total_slices = 0
        
        print(f"{len(self.paths)}개의 NIfTI 파일을 찾았습니다. 슬라이스 인덱싱 중...")
        for path in self.paths:
            try:
                img_shape = nib.load(path).shape
                # 슬라이스가 마지막 축에 있다고 가정 (예: W, H, Slices)
                num_slices = img_shape[2]
                self.slice_map.append({'path': path, 'num_slices': num_slices})
                total_slices += num_slices
                self.cumulative_slices.append(total_slices)
            except Exception as e:
                print(f"{path} 파일을 읽는 데 실패하여 건너뜁니다. 오류: {e}")
        
        print(f"인덱싱된 총 슬라이스 수: {total_slices}")
        self.volume_cache = {} # 로드된 NIfTI 볼륨을 캐싱하여 속도 향상

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, idx):
        # 전체 인덱스(idx)를 파일 인덱스와 파일 내 슬라이스 인덱스로 변환
        file_idx = np.searchsorted(self.cumulative_slices, idx, side='right') - 1
        slice_in_file = idx - self.cumulative_slices[file_idx]

        file_path = self.slice_map[file_idx]['path']

        # 캐시 또는 디스크에서 볼륨 데이터 로드
        if file_path in self.volume_cache:
            volume_data = self.volume_cache[file_path]
        else:
            volume_data = nib.load(file_path).get_fdata()
            self.volume_cache[file_path] = volume_data
            
        # 슬라이스 추출
        img = volume_data[:, :, slice_in_file].astype(np.float32)
        
        # 정규화
        vmin, vmax = np.min(img), np.max(img)
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)

        # 채널 차원 추가: (H, W) -> (1, H, W)
        img = np.expand_dims(img, 0)
        x = torch.from_numpy(img.copy())

        # 학습 스크립트의 데이터 형식을 맞추기 위한 더미(dummy) 값들
        # 실제 사이노그램은 학습 루프에서 생성되므로 여기서는 placeholder만 반환
        dummy_sino = torch.zeros(720, 768) 
        dummy_pos_data = torch.zeros_like(dummy_sino)
        dummy_max_val = torch.tensor(1.0)
        dummy_min_val = torch.tensor(0.0)
        dummy_label = torch.zeros_like(x) # 레이블은 사용하지 않음

        fname = f"{os.path.basename(file_path)}_slice_{slice_in_file}"

        # 반환 형식: (이미지, 사이노그램, 위치데이터, 최대값, 최소값, 사이노그램레이블, 이미지레이블, 파일명)
        return x, dummy_sino, dummy_pos_data, dummy_max_val, dummy_min_val, dummy_sino, dummy_label, fname
    
class AAPMDataLoader:
    # def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, train_val="train", augment=True):
    #     """
    #     Args:
    #         dataset (Dataset): Custom dataset class instance (AAPMDataset).
    #         batch_size (int): Number of samples per batch.
    #         shuffle (bool): If True, shuffle the data.
    #         num_workers (int): Number of subprocesses to use for data loading.
    #     """
    #     if dataset.dataset_name == "udpet":
    #         self.dataset_path = dataset.dataset_path
    #         if train_val == "train":
    #             self.ids = dataset.train_set
    #             # Initialize the dataset and DataLoader
    #             self.dataset = AAPMDataset(self.dataset_path, self.ids, bank_path="./pos_data/prior_sino.npy",augment=False)
    #         else:
    #             self.ids = dataset.val_set
    #             # Initialize the dataset and DataLoader
    #             self.dataset = AAPMDataset(self.dataset_path, self.ids, bank_path="./pos_data/prior_sino.npy",augment=False)

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, train_val="train", augment=True):
        if dataset.dataset_name == "udpet":
            base = dataset.dataset_path
            # train/val 구분이 필요하면 여기서 하위 폴더를 나눠 지정
            self.dataset = UDPETDataset(base)
        else:
            # 기존 npy용 AAPMDataset 초기화 로직 유지
            self.dataset = AAPMDataset(dataset.dataset_path, dataset.train_set if train_val=="train" else dataset.val_set,
                                       bank_path="./pos_data/prior_sino.npy", augment=False)
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty.")
        # [수정] 학습 모드일 때만 drop_last=True 설정
        # 마지막 배치가 1개 남으면 버려서 BatchNorm 에러 방지
        should_drop_last = (train_val == "train")
        
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=should_drop_last)

    def get_loader(self):
        return self.dataloader

# 추가: 재귀 수집 함수
def collect_dicom_paths(root_dir: str):
    pats = []
    for ext in ("*.IMA", "*.ima", "*.dcm", "*.DCM"):
        # root_dir 아래 모든 하위 디렉토리를 재귀적으로 뒤져라
        pats += glob(os.path.join(root_dir, "**", ext), recursive=True)
    return sorted(pats)


# 추가: DICOM → HU 변환
def dicom_to_hu(ds):
    arr = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1))
    inter = float(getattr(ds, "RescaleIntercept", 0))
    return (arr * slope + inter).astype(np.float32)

# 신규 데이터셋: UDPET DICOM 슬라이스
class UDPETDataset(Dataset):
    def __init__(self, base_dir: str):
        self.paths = collect_dicom_paths(base_dir)
        if len(self.paths) == 0:
            raise RuntimeError(f"No DICOM found under: {base_dir}")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        ds = pydicom.dcmread(p)
        img = dicom_to_hu(ds)                      # (H, W), float32
        # [선택] 정규화
        vmin, vmax = np.min(img), np.max(img)
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        img = np.expand_dims(img, 0)               # (1, H, W)
        x = torch.from_numpy(img)                  # torch.float32
        # 현재 라벨 없음 → 더미 0
        y = torch.tensor(0)
        fname = os.path.basename(p)
        return x, y, fname
