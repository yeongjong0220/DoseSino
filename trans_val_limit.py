from model.transsino import Transformer
import torch
from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os
from dataset.data_loader import AAPMDataLoader
from utils.transforms import radon_fanbeam
from torch_radon import Radon
from utils.metrics import cal_psnr_torch, cal_psnr_np, gradient_loss, calculate_metric_percase
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import math
from torchvision.utils import save_image
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer for sinogram view synthesis")
    parser.add_argument("--config", default='path/to/config', type=str, 
                        help="Path to the config file")
    parser.add_argument("--ckpt", default='ckpt/', type=str, 
                        help="Path to the checkpoint file for loading the model")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="Number of classes")
    parser.add_argument("--vit_name", default='R50-ViT-B_16', type=str,
                        help='select one vit model')
    parser.add_argument("--n_skip", default=3, type=int,
                        help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', default=16, type=int,
                        help='vit_patches_size, default is 16')  
    parser.add_argument("--angles", default=120, type=int,
                        help="Number of views for sinogram")
    
    args = parser.parse_args()
    
    return args


class ModelArgs:
    dim: int = 352
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: int = 32
    vocab_size: int = 768
    multiple_of: int = 1024
    ffn_dim_multiplier: float = 1.3
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 6
    max_seq_len: int = 720
model_parallel_size = None


class DotDict(dict):
    """Dot notation access to dictionary attributes, incluidng nested dictionaries."""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        return self.get(attr)
    
    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self[key] = value
        
    def __delattr__(self, key):
        del self[key]


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return DotDict(config)


def ssim_torch(img1, img2, window_size=11, sigma=1.5, data_range=1.0, K1=0.01, K2=0.03):
    if img1.dim() == 3:
        img1 = img1.unsqueeze(1)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(1)

    B, C, H, W = img1.shape
    pad = window_size // 2

    # Gaussian window (2D)
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype) - pad
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-8)
    window2d = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    window2d = window2d.repeat(C, 1, 1, 1)  # [C,1,ws,ws]

    mu1 = F.conv2d(img1, window2d, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window2d, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window2d, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window2d, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window2d, padding=pad, groups=C) - mu12

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    return ssim_map.mean()  # scalar

def calculate_psnr(img1, img2):
    """Calculate PSNR given two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100 
    max_pixel_value = 1.0
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

def radon(sinogram, num_view=352):
    image_size = 300
    detector_count = 352
    angles = torch.linspace(0, np.pi, num_view, device='cuda')
    
    radon = Radon(
        image_size,
        det_count=detector_count,
        angles=angles,
    )
    
    filterd_sino = radon.filter_sinogram(sinogram)
    recon_img = radon.backprojection(filterd_sino)
    
    return recon_img

# Example usage:
if __name__ == "__main__":
    config = load_config("config/data_config.yaml")
    
    args = parse_args()
    
    model_args: ModelArgs = ModelArgs()
    vit_configs = CONFIGS_ViT_seg[args.vit_name]
    vit_configs.n_classes = args.num_classes
    vit_configs.n_skip = args.n_skip    
    vit_configs.patches.grid = (int(352 / args.vit_patches_size), int(352 / args.vit_patches_size))
    vit_configs.img_size = (352, 352)    
    transformer = Transformer(model_args,vit_configs).to("cuda")
    start_pos = 0
    #checkpoint = torch.load(args.ckpt)
    os.makedirs("val_op_val2", exist_ok=True)
    os.makedirs("val_masked_seg", exist_ok=True)
    os.makedirs("val_figure", exist_ok=True)

    if os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt)
        if 'learnable_prior_patterns' in checkpoint:
            ckpt_param = checkpoint['learnable_prior_patterns']
            model_shape = transformer.learnable_prior_patterns.shape # [1, 400, 352, 352]
            
            if ckpt_param.dim() == 3:
                print(f"[Info] Reshaping 'learnable_prior_patterns' from {ckpt_param.shape} to {model_shape}")
                checkpoint['learnable_prior_patterns'] = ckpt_param.unsqueeze(0)
                
        # ------------------------------------------------
        transformer.load_state_dict(checkpoint, strict=False)
        print(f"Model weights loaded from {args.ckpt}")

    val_loader = AAPMDataLoader(dataset=config.aapm_dataset, batch_size=1, shuffle=True, num_workers=0, train_val="val").get_loader()
    total_psnr = 0
    num_images = 0
    t_loss_val = 0
    t_gloss_val = 0
    t_l2loss_val = 0
    total_psnr_val = 0
    num_images_val = 0
    total_ssim = 0.0
    count = 0

    i = 0
    
    X, y = [], []
    transformer.eval()

    with torch.no_grad():
        for img_val, full_sino_val, input_sino_val, max_value_val, min_value_val, sino_label_val, label_val, file_name, dose_val in tqdm(val_loader):
            full_sino_val = full_sino_val.to("cuda")
            img_val = img_val.to("cuda")
            max_value_val = max_value_val.to("cuda")
            min_value_val = min_value_val.to('cuda')
            # pos_data_val = pos_data_val.to(device="cuda", dtype=img_val.dtype)
            input_sino_val = input_sino_val.to('cuda')
            label_val = label_val.to('cuda')
            
            op_val, op_val2, seg_val, _, attn_score, pred_dose_val = transformer.forward(input_sino_val, start_pos, min_value_val, max_value_val)

            dv_op_val = torch.clamp(op_val, 0, 1)
            gt_img = torch.clamp(img_val, 0, 1)

            psnr_val = calculate_psnr(dv_op_val, gt_img)          # torch scalar
            ssim_val = ssim_torch(dv_op_val, gt_img).item()       # float

            total_psnr += float(psnr_val)
            total_ssim += float(ssim_val)
            count += 1

            low_dose_image_val = radon(input_sino_val)
            psnr_low_val = calculate_psnr(img_val, low_dose_image_val)
            
            seg_iradon = radon(seg_val)
            low_dose_image_val = low_dose_image_val.squeeze(0).detach().cpu().numpy()
            
            dv_op_val = dv_op_val.squeeze(0).detach().cpu().numpy()
            img_val = img_val.squeeze(0).detach().cpu().numpy()
            input_sino_val = input_sino_val.squeeze(0).detach().cpu().numpy()
            op_val2 = op_val2.squeeze(0).detach().cpu().numpy()
            full_sino_val = full_sino_val.squeeze(0).detach().cpu().numpy()
            seg_val = torch.argmax(seg_iradon, dim=1).squeeze(0).detach().cpu().numpy()
            label_val = label_val.squeeze(0).detach().cpu().numpy()
            
            cur_dice, cur_hd95 = calculate_metric_percase(seg_val == 1, label_val == 1)
            
            file_stem = os.path.splitext(file_name[0])[0]

            fn_op  = f"{file_stem}_psnr{psnr_val:.2f}_ssim{ssim_val:.4f}_op_val2.png"
            fn_msk = f"{file_stem}_psnr{psnr_val:.2f}_ssim{ssim_val:.4f}_mask.png"

            plt.imsave(os.path.join("val_op_val2", fn_op), dv_op_val, cmap="gray", vmin=0, vmax=1)

            mask_u8 = (seg_val != 0).astype(np.uint8) * 255
            plt.imsave(os.path.join("val_masked_seg", fn_msk), mask_u8, cmap="gray", vmin=0, vmax=255)
            # ----------------------------------------


            plt.figure(figsize=(20,5))
            plt.subplot(1, 6, 1)
            plt.imshow(img_val, cmap='gray')
            masked_label = np.ma.masked_where(label_val == 0, label_val)
            plt.imshow(masked_label, cmap='jet', alpha=0.6)
            plt.title("original image"); plt.axis('off')
            
            plt.subplot(1, 6, 2)
            plt.imshow(full_sino_val, cmap='gray')
            plt.title("full sino"); plt.axis('off')
            
            plt.subplot(1, 6, 3)
            plt.imshow(input_sino_val, cmap='gray')
            plt.title("low dose sino"); plt.axis('off')
            
            plt.subplot(1, 6, 4)
            plt.imshow(low_dose_image_val, cmap='gray')
            plt.title(f"low dose image (psnr : {psnr_low_val:.4f})"); plt.axis('off')
            
            plt.subplot(1, 6, 5)
            plt.imshow(op_val2, cmap='gray')
            plt.title("recon sino"); plt.axis('off')
            
            plt.subplot(1, 6, 6)
            plt.imshow(dv_op_val, cmap='gray')
            masked_seg = np.ma.masked_where(seg_val == 0, seg_val)
            plt.imshow(masked_seg, cmap='jet', alpha=0.6)
            plt.title(f"Recon (PSNR: {psnr_val:.2f})\nDice: {cur_dice:.4f}, HD: {cur_hd95:.2f}", fontsize=10); plt.axis('off')
            
            file_name = file_name[0]
            file_name = file_name[:-4]
            
            graph_path = f"val_figure/{file_name}.png"
            i = i + 1
            
            plt.savefig(graph_path)
            plt.close()
        
        mean_psnr = total_psnr / max(count, 1)
        mean_ssim = total_ssim / max(count, 1)
        print(f"[VAL] Mean PSNR: {mean_psnr:.4f} | Mean SSIM: {mean_ssim:.6f} | N={count}")
