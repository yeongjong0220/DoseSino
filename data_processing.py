import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import math
from tqdm import tqdm
from torch_radon import Radon
from wavelet2 import osem_reconstruct


def load_nifti(path):
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    return arr


def pet_to_label_name(pet_name):
    base = pet_name.replace(".nii.gz", "")
    # 마지막 _0001 만 제거
    if base.endswith("_0001"):
        base = base[:-5]
    return base + ".nii.gz"


def run_split(root_dir, split, device, n_angles, det_count):

    image_dir = root_dir / split / "image"
    label_dir = root_dir / split / "label"

    pet_files = sorted(image_dir.glob("*.nii.gz"))
    label_files = sorted(label_dir.glob("*.nii.gz"))


    half_idx = len(pet_files) 
    pet_files = pet_files[:half_idx]
    
    seg_dict = {s.name.replace(".nii.gz", ""): s for s in label_files}

    print(f"[INFO] {split}: Found {len(pet_files)} PET files")

    for pet_path in tqdm(pet_files, desc=f"Processing {split}"):

        lbl_file = pet_to_label_name(pet_path.name)
        lbl_key = lbl_file.replace(".nii.gz", "")

        if lbl_key not in seg_dict:
            print("[WARN] Missing label for:", pet_path.name)
            continue

        seg_path = seg_dict[lbl_key]

        process_study(
            pet_path, seg_path,
            root_dir / split,
            device,
            n_angles, det_count
        )

def process_study(pet_path, seg_path, out_dir, device, n_angles, det_count):

    stem = pet_path.name.replace(".nii.gz", "")

    try:
        pet = load_nifti(pet_path)
    except (EOFError, OSError, nib.filebasedimages.ImageFileError) as e:
        print(f"\n[WARNING] 손상된 파일 발견! 건너뜁니다: {pet_path}")
        print(f"  Error detail: {e}")
        return # 이 파일(study)은 처리를 중단하고 다음으로 넘어감
    seg = load_nifti(seg_path)
    H, W, Z = pet.shape

    angles = torch.linspace(0, math.pi, steps=n_angles, device=device)
    radon = Radon(
        resolution=300,
        angles=angles,
        det_count=det_count,
        clip_to_circle=True
    )

    # create folders
    inp_dir = out_dir / "input_img-ref"
    org_dir = out_dir / "original_img-ref"
    lbl_dir = out_dir / "labels_img-ref"

    inp_dir.mkdir(parents=True, exist_ok=True)
    org_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 3D label 저장
    #np.save(lbl_dir / f"{stem}_label.npy", seg)

    for z in range(Z):

        if np.max(seg[:, :, z]) == 0:
            continue

        img2d = pet[:, :, z][50:-50, 50:-50]

        if img2d.max() > img2d.min():
            img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min())

        img_t = torch.tensor(img2d, dtype=torch.float32, device=device)
        sino_full = radon.forward(img_t)

        sino_1 = torch.poisson(sino_full * 0.01) / 0.01
        sino_5 = torch.poisson(sino_full * 0.05) / 0.05
        sino_10 = torch.poisson(sino_full * 0.10) / 0.10
        sino_25 = torch.poisson(sino_full * 0.25) / 0.25
        sino_50 = torch.poisson(sino_full * 0.50) / 0.50
        
        prefix = f"{stem}_{z:03d}"
        rec1 = osem_reconstruct(sino_1)
        rec5 = osem_reconstruct(sino_5)
        rec10 = osem_reconstruct(sino_10)
        rec25 = osem_reconstruct(sino_25)
        rec50 = osem_reconstruct(sino_50)
        
        # ✅ [추가] 저장 전 정규화(최소 수정)
        def norm01(rec, eps=1e-8):
            rec = torch.clamp(rec, min=0.0)          # 음수 제거
            m = rec.max()
            if torch.isfinite(m) and m > 0:
                rec = rec / (m + eps)                # [0,1]로 스케일
            return rec

        rec1  = norm01(rec1)
        rec5  = norm01(rec5)
        rec10 = norm01(rec10)
        rec25 = norm01(rec25)
        rec50 = norm01(rec50)

        np.save(inp_dir / f"{prefix}_rec_1.npy", rec1.detach().cpu().numpy())
        np.save(inp_dir / f"{prefix}_rec_5.npy", rec5.detach().cpu().numpy())
        np.save(inp_dir / f"{prefix}_rec_10.npy", rec10.detach().cpu().numpy())
        np.save(inp_dir / f"{prefix}_rec_25.npy", rec25.detach().cpu().numpy())
        np.save(inp_dir / f"{prefix}_rec_50.npy", rec50.detach().cpu().numpy())

        sino_1 = radon.forward(rec1).squeeze(0)
        sino_5 = radon.forward(rec5).squeeze(0)
        sino_10 = radon.forward(rec10).squeeze(0)
        sino_25 = radon.forward(rec25).squeeze(0)
        sino_50 = radon.forward(rec50).squeeze(0)
        
        prefix = f"{stem}_{z:03d}"

        np.save(org_dir / f"{prefix}_image.npy", img2d)
        np.save(org_dir / f"{prefix}_sino_full.npy", sino_full.cpu().numpy())

        np.save(inp_dir / f"{prefix}_sino_1.npy", sino_1.cpu().numpy())
        np.save(inp_dir / f"{prefix}_sino_5.npy", sino_5.cpu().numpy())
        np.save(inp_dir / f"{prefix}_sino_10.npy", sino_10.cpu().numpy())
        np.save(inp_dir / f"{prefix}_sino_25.npy", sino_25.cpu().numpy())
        np.save(inp_dir / f"{prefix}_sino_50.npy", sino_50.cpu().numpy())
 

        np.save(lbl_dir / f"{prefix}_label_slice.npy", seg[:, :, z])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)   # = datasets
    ap.add_argument("--n_angles", type=int, default=352)
    ap.add_argument("--det_count", type=int, default=352)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    root = Path(args.in_dir)

    #run_split(root, "train", device, args.n_angles, args.det_count)
    run_split(root, "val", device, args.n_angles, args.det_count)

    print("[DONE]")


if __name__ == "__main__":
    main()




