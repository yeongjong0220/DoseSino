from model.transsino import Transformer
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from utils.seg_utils import DiceLoss
import os
from dataset.data_loader import AAPMDataLoader
from utils.transforms import radon_fanbeam
from utils.metrics import gradient_loss, calculate_metric_percase, CAFL
from utils.perceptual_loss import VGGPerceptualLoss as p_loss
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from utils import VGGPerceptualLoss as vgg
import argparse
import math
from torchvision.utils import save_image
import matplotlib
from torch_radon import Radon
from utils.custom_losses import VGGPerceptualLoss, FrequencyLoss 
matplotlib.use('Agg')

torch.autograd.set_detect_anomaly(True) 
def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer for sinogram view synthesis")
    parser.add_argument("--config", default='path/to/config', type=str, 
                        help="Path to the config file")
    parser.add_argument("--ckpt", default='ckpt/', type=str,
                        help="Path to the checkpoint file for loading the model")
    parser.add_argument("--log", default='log', type=str,
                        help="Path to the log file")
    parser.add_argument("--log_val", default='logval', type=str,
                        help="Path to the log file for validation")    
    parser.add_argument("--save_dir", default='ckpt', type=str,
                        help="Path to the save directory")
    parser.add_argument("--fig_save_dir", default='fig', type=str,
                        help="Path to the training loss graph save directory")
    
    parser.add_argument("--epochs", default=400, type=int,
                       help="Number of epochs for training")
    parser.add_argument("--lr", default=1e-4, type=float,#1e-4
                        help="Learning rate for training")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="Number of classes")
    parser.add_argument("--vit_name", default='R50-ViT-B_16', type=str,
                        help='select one vit model')
    parser.add_argument("--n_skip", default=3, type=int,
                        help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', default=16, type=int,
                        help='vit_patches_size, default is 16')
    parser.add_argument("--views", default=352, type=int,
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

    max_batch_size: int = 2
    max_seq_len: int = 720
model_parallel_size = None



class DotDict(dict):
    """Dot notation access to dictionary attributes, including nested dictionaries."""
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

if __name__ == "__main__":
    config = load_config("config/data_config.yaml")
    
    args = parse_args()
    
    model_args: ModelArgs = ModelArgs()
    vit_configs = CONFIGS_ViT_seg[args.vit_name]
    vit_configs.n_classes = args.num_classes
    vit_configs.n_skip = args.n_skip    
    vit_configs.patches.grid = (int(352 / args.vit_patches_size), int(352 / args.vit_patches_size))
    vit_configs.img_size = (352, 352)
    
    transformer = Transformer(model_args, vit_configs).to("cuda")
    start_pos = 0 

    print(f"Loading checkpoint from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt)
    
    model_state_dict = transformer.state_dict()
    transformer.load_state_dict(model_state_dict, strict=False)

    
    # Loss
    criterion_vgg = VGGPerceptualLoss().cuda()
    criterion_freq = FrequencyLoss().cuda()
    l2loss = torch.nn.MSELoss().to("cuda")
    class_weights = torch.tensor([1.0, 1.2]).to("cuda") 
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(args.num_classes)    
    criterion_dose = nn.SmoothL1Loss().cuda()
    
    prior_params = []
    other_params = []

    for name, param in transformer.named_parameters():
        if "learnable_prior_patterns" in name:
            prior_params.append(param)
            print(f"Detected Prior Parameter: {name}") =
        else: 
            other_params.append(param)

    optim = torch.optim.Adam([
        {'params': other_params, 'lr': args.lr},            
        {'params': prior_params, 'lr': args.lr * 0.5}     
    ])


    train_loader = AAPMDataLoader(dataset=config.aapm_dataset, batch_size=4, shuffle=True, num_workers=4, augment=False, train_val="train").get_loader()
    val_loader = AAPMDataLoader(dataset=config.aapm_dataset, batch_size=1, shuffle=False, num_workers=4, train_val="val", augment=False).get_loader()
    total_psnr = 0
    num_images = 0
    loss_history = []
    loss_val_history = []

    for epoch in range(0, args.epochs):   
        print('start epoch: ', epoch)
        epoch_loss = 0
        #t_gloss = 0
        t_l2loss = 0
        #t_fbploss = 0
        t_segloss = 0
        epoch_dose_mae = 0.0 =
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for img, full_sino, input_sino, max_value, min_value, sino_label, label, _, dose in pbar:
                max_value = max_value.to("cuda")
                min_value = min_value.to('cuda')
                img = img.to("cuda")
                sino = full_sino.to("cuda")
                input_sino = input_sino.to('cuda')
                sino_label = sino_label.to("cuda")
                label = label.to("cuda")
                target_dose = dose.to("cuda").float()
                target_dose_log = torch.log1p(target_dose)
                op, op2, seg_results, caf, attn_score, pred_dose = transformer.forward(input_sino, start_pos, min_value, max_value) 

                if pred_dose is not None:
                    with torch.no_grad():
                        pred_real = torch.expm1(pred_dose.view(-1))
                        pred_real = torch.clamp(pred_real, 0.0, 100.0)
                        mae = torch.abs(pred_real - target_dose.view(-1)).mean().item()
                        epoch_dose_mae += mae

                B, C, H, W = seg_results.shape
                seg_reshaped = seg_results.view(B * C, H, W) 
                
                # 2. Radon backward
                seg_img_reshaped = radon(seg_reshaped) 
                
                new_H, new_W = seg_img_reshaped.shape[-2], seg_img_reshaped.shape[-1]
                seg_iradon = seg_img_reshaped.view(B, C, new_H, new_W)
                loss_ce = ce_loss(seg_iradon, label[:].long())
                loss_dice = dice_loss(seg_iradon, label, softmax=True)
                seg_loss = 0.5 * loss_ce + 0.7 * loss_dice
                loss_l2 = l2loss(op2, sino) 

                op_min = op.amin(dim=(-2, -1), keepdim=True)
                op_max = op.amax(dim=(-2, -1), keepdim=True)
                epsilon = 1e-8
                range_op = op_max - op_min + epsilon
                
                recon_norm = torch.where(
                    range_op < epsilon,
                    torch.zeros_like(op),
                    (op - op_min) / range_op
                )
                loss_fbp_l2 = l2loss(recon_norm, img)

                if pred_dose is not None:
                    loss_dose_val = criterion_dose(pred_dose.view(-1), target_dose_log.view(-1))
                else:
                    loss_dose_val = 0.0

                loss_low, loss_high = criterion_freq(recon_norm, img)
                loss_freq_total = loss_low + loss_high
                
                loss_vgg_val = criterion_vgg(recon_norm, img)
                
                loss_t = (
                      loss_l2 * 1.0             # Sinogram L2
                    + loss_fbp_l2 * 1.0         # Image L2
                    + seg_loss * 1.0            # Segmentation Loss 
                    + (loss_dose_val * 0.1)     
                    + loss_freq_total * 0.5              #  Freq Loss (Detail & Structure)
                    + loss_vgg_val * 0.001               #  Perceptual Loss
                )

                epoch_loss += loss_t.item()
                t_l2loss += loss_l2.item()

                optim.zero_grad()
                loss_t.backward()
                optim.step()
                pbar.set_postfix({"Loss": loss_t.item()})
                pbar.update(1)
                
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss:.4f}")
        
        save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(transformer.state_dict(), save_path)
        print(f"Model weights saved at {save_path}")
        
        # --- Validation Loop ---
        transformer.eval()
        t_loss_val = 0
        t_l2loss_val = 0
        t_dose_mae_val = 0.0 

        total_psnr_val = 0
        num_images_val = 0
        metric_list = 0.0

        with torch.no_grad():
            for img_val, sino_val, input_sino_val, max_value_val, min_value_val, sino_label_val, label_val, file_name, dose_val in tqdm(val_loader):
                sino_val = sino_val.to("cuda")
                input_sino_val = input_sino_val.to('cuda')
                img_val = img_val.to("cuda")
                max_value_val = max_value_val.to("cuda")
                min_value_val = min_value_val.to('cuda')
                sino_label_val = sino_label_val.to("cuda")
                label_val = label_val.to("cuda")
                dose_val = dose_val.to("cuda").float() 
.
                op_val, op_val2, seg_val, _, attn_score, pred_dose_val= transformer.forward(input_sino_val, start_pos, min_value_val, max_value_val)
                seg_iradon = radon(seg_val)
                loss_ce = ce_loss(seg_iradon, label_val[:].long())
                loss_dice = dice_loss(seg_iradon, label_val, softmax=True)
                seg_loss_val = 0.5 * loss_ce + 0.5 * loss_dice                 
                label_val = label_val.squeeze(0)
                seg_out_val = torch.argmax(torch.softmax(radon(seg_val), dim=1), dim=1).squeeze(0)

                prediction = seg_out_val.cpu().detach().numpy()
                metric_i = []
                for i in range(1, vit_configs.n_classes):
                    metric_i.append(calculate_metric_percase(prediction == i, label_val.cpu().detach().numpy() == i))
                metric_list += np.array(metric_i)
                
                l2loss_val = l2loss(op_val2, sino_val)

                op_min_val = op_val.amin(dim=(-2, -1), keepdim=True)
                op_max_val = op_val.amax(dim=(-2, -1), keepdim=True)
                range_op_val = op_max_val - op_min_val
                epsilon = 1e-8
                
                recon_norm_val = torch.where(
                    range_op_val < epsilon,
                    torch.zeros_like(op_val),
                    (op_val - op_min_val) / range_op_val
                )
                
                loss_fbp_l2_val = l2loss(recon_norm_val, img_val)
                loss_t_val = l2loss_val + loss_fbp_l2_val + seg_loss_val # + gloss_val + loss_fbp_gradient_val
                t_l2loss_val += l2loss_val.item()
                t_loss_val += loss_t_val.item()
                
                if pred_dose_val is not None:
                    # Log Scale -> Real Scale 
                    pred_real = torch.expm1(pred_dose_val.view(-1))
                    pred_real = torch.clamp(pred_real, 0.0, 100.0) # 0~100% cliping
                    
                    # Absolute Error
                    mae = torch.abs(pred_real - dose_val.view(-1)).mean().item()
                    t_dose_mae_val += mae

                dv_op_val = torch.clamp(op_val, 0, 1)
                
                total_psnr_val += calculate_psnr(dv_op_val, img_val)
                num_images_val += 1
            
            avg_val_loss = t_loss_val / len(val_loader)
            avg_dose_mae = t_dose_mae_val / len(val_loader) 
            loss_val_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {avg_val_loss:.4f}, Val Dose MAE: {avg_dose_mae:.2f}%")
        
        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='red', label='training loss')
        plt.plot(range(1, len(loss_val_history) + 1), loss_val_history, marker='s', color='blue', label='val_loss')
        plt.title(f"{epoch+1} Epoch Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        graph_path = os.path.join(args.fig_save_dir, f"loss_epoch.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"Loss graph saved at {graph_path}")        
        
        print(f"psnr : {total_psnr_val / len(val_loader)}")
        metric_list = metric_list / len(val_loader)
        for i in range(1, vit_configs.n_classes):
            print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        transformer.train()
