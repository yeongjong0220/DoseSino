# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# from pathlib import Path
# import math
# import traceback
# from tqdm import tqdm
# import nibabel as nib
# import numpy as np
# import torch
# from torch_radon import Radon  # parallel-beam Radon
# from wavelet2 import osem_reconstruct


# def load_nifti(path, canonical=True):
#     """
#     NIfTI 로드 (옵션: RAS 기준으로 정렬)
#     반환: vol[np.float32, (X,Y,Z)], axcodes(tuple)
#     """
#     img = nib.load(str(path))
#     img = nib.as_closest_canonical(img)
#     vol = img.get_fdata(dtype=np.float32)  # (X, Y, Z)
#     vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
#     vol = np.clip(vol, 0.0, None).astype(np.float32, copy=False)
#     # vol = resize_depth_legs(vol)
#     ax = nib.aff2axcodes(img.affine)       # ('R','A','S') 예상
#     return vol, ax

# def resize_depth_legs(img, target_depth=320):
#     """
#     img: numpy array of shape (H, W, D)
#     target_depth: 원하는 depth 크기
#     """
#     h, w, d = img.shape
#     if d == target_depth:
#         return img

#     if d > target_depth:
#         # 앞부분(다리 쪽) slice를 버리고, 뒤쪽(머리 쪽)만 보존
#         return img[:, :, d - target_depth:]
#     else:
#         # 부족하면 앞부분(다리 쪽)에 padding 추가
        
#         return img

# def find_nifti_files(in_dir, recursive=True):
#     """
#     입력 폴더에서 .nii, .nii.gz 파일 목록 찾기 (정렬)
#     """
#     in_dir = Path(in_dir)
#     files = []

#     files += list(in_dir.glob("*.nii.gz"))

#     # 중복 제거 + 정렬
#     files = sorted(set(files))
#     return files


# def process_one_file(nifti_path: Path,
#                      out_root: Path,
#                      n_angles: int,
#                      det_count_arg: int,
#                      device: torch.device,
#                      ):
#     """
#     단일 NIfTI 파일 처리:
#     - 슬라이스 .npy 저장
#     - 슬라이스별 Radon 시노그램 .npy 저장
#     """
#     # 출력 디렉토리 준비: <out_root>/<파일이름without_ext>/slices, sinos
#     # .nii.gz 고려해서 stem을 안전하게 만듦
#     stem = nifti_path.name
#     if stem.endswith(".nii.gz"):
#         stem = stem[:-7]
#     elif stem.endswith(".nii"):
#         stem = stem[:-4]
#     file_out_dir = out_root
#     original_dir = file_out_dir / "original"
#     output_dir = file_out_dir / "input"
#     original_dir.mkdir(parents=True, exist_ok=True)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 1) 로드
#     vol, ax = load_nifti(nifti_path)  # (X,Y,Z)
#     X, Y, Z = vol.shape
#     print(f"[{nifti_path.name}] shape=(X={X}, Y={Y}, Z={Z}), ax={ax}")

#     # 2) 각도 텐서
#     angles = torch.linspace(0.0, math.pi, steps=n_angles, dtype=torch.float32, device=device)

#     # 3) 첫 슬라이스로 Radon 연산자 설정 (정사각 크기/검출기 수)
#     sample_slice = vol[:, :, 0].astype(np.float32)
#     img_size = int(sample_slice.shape[0])
#     det_count = det_count_arg if det_count_arg is not None else img_size

#     radon = Radon(resolution=300,
#                   angles=angles,
#                   det_count=det_count,
#                   clip_to_circle=True)

#     # 한 슬라이스만 처리
    

#     with torch.no_grad():
#         for z in range(Z // 2, Z):
#             img2d = vol[:,:,z]
#             img2d = img2d[50:-50, 50:-50]
#             if img2d.max() != img2d.min():
#                 img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min())
#             print(img2d.shape)
#             # 원본 슬라이스 저장
#             np.save(original_dir / f"{stem}_{z:03d}_image.npy", img2d)
            
#             img_t = torch.from_numpy(img2d).to(device)
            
#             # Radon
#             sino = radon.forward(img_t)
#             sino_1 = torch.poisson(sino * 0.01) / 0.01
#             sino_10 = torch.poisson(sino * 0.1) / 0.1
#             sino_25 = torch.poisson(sino * 0.25) / 0.25
            
#             recon1 = osem_reconstruct(sino_1)
#             recon10 = osem_reconstruct(sino_10)
#             recon25 = osem_reconstruct(sino_25)
            
#             sino_1 = radon.forward(recon1).squeeze(0)
#             sino_10 = radon.forward(recon10).squeeze(0)
#             sino_25 = radon.forward(recon25).squeeze(0)
            
#             sino_np = sino.detach().cpu().numpy()
#             sino_1_np = sino_1.detach().cpu().numpy()
#             sino_10_np = sino_10.detach().cpu().numpy()
#             sino_25_np = sino_25.detach().cpu().numpy()
            
#             print(f"sino shape : {sino_1.shape}")
#             print(f"img scale : {img_t.min()} ~ {img_t.max()}, sino scale : {sino_np.min()} ~ {sino_np.max()}")
            
#             np.save(original_dir / f"{stem}_{z:03d}_sino_full.npy", sino_np)
#             np.save(output_dir / f"{stem}_{z:03d}_sino_1", sino_1_np)
#             np.save(output_dir / f"{stem}_{z:03d}_sino_10", sino_10_np)
#             np.save(output_dir / f"{stem}_{z:03d}_sino_25", sino_25_np)
#             print(f"  - {nifti_path.name}: done")
#     # 4) 모든 슬라이스 처리
#     # with torch.no_grad():
#     #     for z in range(Z):
#     #         img2d = vol[:, :, z].astype(np.float32)                # (H,W)
#     #         # 원본 슬라이스 저장(패딩 없이)
#     #         np.save(slices_dir / f"{stem}_slice_{z:03d}.npy", img2d)

#     #         # 패딩 후 텐서로 변환 (B=1, C=1)
#     #         img_t = torch.from_numpy(img2d).to(device)  # (1,1,S,S)

#     #         # Radon (B, n_angles, det_count)
#     #         sino = radon.forward(img_t)
#     #         sino_np = sino[0].detach().cpu().numpy()               # (n_angles, det_count)

#     #         # 시노그램 저장
#     #         np.save(sinos_dir / f"{stem}_sino_{z:03d}.npy", sino_np)

#     #         if (z % 10) == 0 or z == Z - 1:
#     #             print(f"  - {nifti_path.name}: [{z+1}/{Z}] done")

#     print(f"[OK] Saved -> {file_out_dir}")


# def main():
#     ap = argparse.ArgumentParser(description="Batch: torch-radon per-slice sinograms for all NIfTI files in a folder")
#     ap.add_argument("--in_dir", required=True, help="Input directory containing .nii / .nii.gz")
#     ap.add_argument("--out_dir", required=True, help="Output root directory")
#     ap.add_argument("--n_angles", type=int, default=352, help="Number of projection angles (0..pi)")
#     ap.add_argument("--det_count", type=int, default=352, help="Number of detectors (default: padded image size)")
#     ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device")
#     args = ap.parse_args()

#     device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
#     print(device)
#     out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

#     files = find_nifti_files(args.in_dir)
#     if not files:
#         print(f"[WARN] No NIfTI files found under: {args.in_dir}")
#         return

#     print(f"[INFO] Found {len(files)} NIfTI files.")
#     for f in tqdm(files):
#         try:
#             process_one_file(
#                 nifti_path=f,
#                 out_root=out_root,
#                 n_angles=args.n_angles,
#                 det_count_arg=args.det_count,
#                 device=device,
#             )
#         except Exception as e:
#             print(f"[ERROR] Failed: {f.name}")
#             traceback.print_exc()

#     print("[DONE] All files processed.")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# python data_processing.py --in_dir D:/dataset/UDPET/Bern-Inselspital-2022/zip/low-dose/datasets
# python data_processing.py --in_dir /mnt/d/dataset/UDPET/Bern-Inselspital-2022/zip/low-dose/datasets
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

    # --- [수정된 부분: 절반만 선택] ---
    # 전체 파일 리스트의 절반까지만 슬라이싱합니다.
    half_idx = len(pet_files) // 2
    pet_files = pet_files[:half_idx]
    # -------------------------------
    
    # 라벨 key dict
    seg_dict = {s.name.replace(".nii.gz", ""): s for s in label_files}

    print(f"[INFO] {split}: Found {len(pet_files)} PET files")

    for pet_path in tqdm(pet_files, desc=f"Processing {split}"):

        # PET → 라벨 파일명 변환
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

    # [수정] try-except로 감싸서 에러가 나면 파일명을 출력하고 건너뜁니다.
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
        # print("sino_1 max,min", sino_1.max(), sino_1.min())
        # print("sino_10 max,min", sino_10.max(), sino_10.min())
        # print("sino_25 max min",sino_25.max(),sino_25.min())
        # print("sino_full max,min",sino_full.max(),sino_full.min())
        # sino_1 max,min tensor(400., device='cuda:0') tensor(0., device='cuda:0')
        # sino_10 max,min tensor(110., device='cuda:0') tensor(0., device='cuda:0')
        # sino_25 max min tensor(76., device='cuda:0') tensor(0., device='cuda:0')
        # sino_full max,min tensor(32.1810, device='cuda:0') tensor(0., device='cuda:0')
        # sino_1 max,min tensor(200., device='cuda:0') tensor(0., device='cuda:0')
        # sino_10 max,min tensor(60., device='cuda:0') tensor(0., device='cuda:0')
        # sino_25 max min tensor(36., device='cuda:0') tensor(0., device='cuda:0')
        # sino_full max,min tensor(13.7018, device='cuda:0') tensor(0., device='cuda:0')
        prefix = f"{stem}_{z:03d}"
        rec1 = osem_reconstruct(sino_1)
        rec5 = osem_reconstruct(sino_5)
        rec10 = osem_reconstruct(sino_10)
        rec25 = osem_reconstruct(sino_25)
        rec50 = osem_reconstruct(sino_50)
        # print("rec1",rec1.min(), rec1.max())
        # print("rec5",rec5.min(), rec5.max())
        # print("rec10",rec10.min(), rec10.max())
        # print("rec25",rec25.min(), rec25.max())
        # print("rec50",rec50.min(), rec50.max())
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

        # sino_1 = radon.forward(rec1).squeeze(0)
        # sino_5 = radon.forward(rec5).squeeze(0)
        # sino_10 = radon.forward(rec10).squeeze(0)
        # sino_25 = radon.forward(rec25).squeeze(0)
        # sino_50 = radon.forward(rec50).squeeze(0)

        # print("sino_1 max,min", sino_1.max(), sino_1.min())      
        # print("sino_10 max,min", sino_10.max(), sino_10.min())
        # print("sino_25 max min",sino_25.max(),sino_25.min())
        # print("sino_full max,min",sino_full.max(),sino_full.min())
        # sino_1 max,min tensor(3.2885, device='cuda:0') tensor(0., device='cuda:0')
        # sino_10 max,min tensor(1.7776, device='cuda:0') tensor(0., device='cuda:0')
        # sino_25 max min tensor(1.6938, device='cuda:0') tensor(0., device='cuda:0')
        # sino_full max,min tensor(23.3695, device='cuda:0') tensor(0., device='cuda:0')
        # sino_1 max,min tensor(3.4093, device='cuda:0') tensor(0., device='cuda:0')
        # sino_10 max,min tensor(1.9221, device='cuda:0') tensor(0., device='cuda:0')
        # sino_25 max min tensor(1.7024, device='cuda:0') tensor(0., device='cuda:0')
        # sino_full max,min tensor(24.2752, device='cuda:0') tensor(0., device='cuda:0')
        
        #prefix = f"{stem}_{z:03d}"

        np.save(org_dir / f"{prefix}_image.npy", img2d)
        #np.save(org_dir / f"{prefix}_sino_full.npy", sino_full.cpu().numpy())

        # np.save(inp_dir / f"{prefix}_sino_1.npy", sino_1.cpu().numpy())
        # np.save(inp_dir / f"{prefix}_sino_5.npy", sino_5.cpu().numpy())
        # np.save(inp_dir / f"{prefix}_sino_10.npy", sino_10.cpu().numpy())
        # np.save(inp_dir / f"{prefix}_sino_25.npy", sino_25.cpu().numpy())
        # np.save(inp_dir / f"{prefix}_sino_50.npy", sino_50.cpu().numpy())
 

        np.save(lbl_dir / f"{prefix}_label_slice.npy", seg[:, :, z])


# def run_split(root_dir, split, device, n_angles, det_count):

#     image_dir = root_dir / split / "image"
#     label_dir = root_dir / split / "label"

#     pet_files = sorted(image_dir.glob("*.nii.gz"))
#     label_files = sorted(label_dir.glob("*.nii.gz"))

#     seg_dict = {}
#     for s in label_files:
#         key = s.name.replace(".nii.gz", "")
#         seg_dict[key] = s

#     print(f"[INFO] {split}: Found {len(pet_files)} PET files")

#     for pet_path in tqdm(pet_files, desc=f"Processing {split}"):

#         lbl_name = pet_to_label_name(pet_path.name).replace(".nii.gz", "")
#         if lbl_name not in seg_dict:
#             print("[WARN] Missing label for:", pet_path.name)
#             continue

#         seg_path = seg_dict[lbl_name]
#         process_study(
#             pet_path, seg_path,
#             root_dir / split,
#             device,
#             n_angles, det_count
#         )


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



