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
        print("MI Data preloaded.")

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
                numbers = re.findall(r'\d+', file_name)
                
                if numbers:
                    slice_idx = int(numbers[-1])
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

    def __getitem__(self, idx):
            sino_path = self.data_file_paths[idx]
            prefix_input = sino_path.split("_sino_")[0]

            # Image (GT): input -> original, suffix -> _image.npy
            image_path = prefix_input.replace("input", "original") + "_image.npy"
            
            # Full Sino (GT): input -> original, suffix -> _sino_full.npy
            full_sino_path = prefix_input.replace("input", "original") + "_sino_full.npy"
            
            # Label (Seg): input -> labels, suffix -> _label_slice.npy
            label_path = prefix_input.replace("input", "labels") + "_label_slice.npy"

            file_name = os.path.basename(sino_path)
            dose_val = 100.0 # default
            
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
                dose_val = 100.0 

            try:
                image = np.array(np.load(image_path))       # 300x300 
                full_sino = np.array(np.load(full_sino_path))
                sino = np.array(np.load(sino_path))
                label = np.array(np.load(label_path))       # 400x400
                
                if label.shape[0] == 400 and label.shape[1] == 400:
                    label = label[50:-50, 50:-50]

                
            except FileNotFoundError as e:
                print(f"\n[Error] Not fuound file: {e.filename}")
                print(f"  - Source: {sino_path}")
                raise e

            if np.max(image) != np.min(image):
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            dose_tensor = torch.tensor([dose_val], dtype=torch.float32)
            image_tensor = torch.tensor(image, dtype=torch.float32)
            full_sino = torch.tensor(full_sino, dtype=torch.float32)
            input_sino = torch.tensor(sino, dtype=torch.float32)
            label = torch.tensor(label)

            if dose_val < 99.0:
                INPUT_NORM_FACTOR = 10.0  
            else:
                INPUT_NORM_FACTOR = 100.0 

            TARGET_NORM_FACTOR = 100.0

            input_sino_norm = torch.clamp(input_sino / full_sino.max(), 0.0, 1.0)
            full_sino_norm = torch.clamp(full_sino / full_sino.max(), 0.0, 1.0)
            
            max_value = torch.tensor(full_sino.max(), dtype=torch.float32)
            min_value = torch.tensor([0.0], dtype=torch.float32)

            max_value = max_value[..., None]
            min_value = min_value[..., None]

            sino_label = torch.tensor(0)
            
            return image_tensor, full_sino_norm, input_sino_norm, max_value, min_value, sino_label, label, file_name, dose_tensor
        
class AAPMDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, train_val="train", augment=True):
        if dataset.dataset_name == "udpet":
            base = dataset.dataset_path
            self.dataset = UDPETDataset(base)
        else:
            self.dataset = AAPMDataset(dataset.dataset_path, dataset.train_set if train_val=="train" else dataset.val_set,
                                       bank_path="./pos_data/prior_sino.npy", augment=False)
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty.")
        should_drop_last = (train_val == "train")
        
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=should_drop_last)

    def get_loader(self):
        return self.dataloader

def collect_dicom_paths(root_dir: str):
    pats = []
    for ext in ("*.IMA", "*.ima", "*.dcm", "*.DCM"):
        pats += glob(os.path.join(root_dir, "**", ext), recursive=True)
    return sorted(pats)

def dicom_to_hu(ds):
    arr = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1))
    inter = float(getattr(ds, "RescaleIntercept", 0))
    return (arr * slope + inter).astype(np.float32)


