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
from utils.custom_losses import VGGPerceptualLoss, FrequencyLoss # [추가]
matplotlib.use('Agg')

torch.autograd.set_detect_anomaly(True)  # NaN/inf 발생 지점 traceback
def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer for sinogram view synthesis")
    parser.add_argument("--config", default='path/to/config', type=str, 
                        help="Path to the config file")
    parser.add_argument("--ckpt", default='ckpt/model_epoch_102.pth', type=str,
                        help="Path to the checkpoint file for loading the model")
    parser.add_argument("--log", default='log', type=str,
                        help="Path to the log file")
    parser.add_argument("--log_val", default='logval', type=str,
                        help="Path to the log file for validation")    
    parser.add_argument("--save_dir", default='ckpt', type=str,
                        help="Path to the save directory")
    parser.add_argument("--fig_save_dir", default='fig', type=str,
                        help="Path to the training loss graph save directory")
    
    parser.add_argument("--epochs", default=150, type=int,
                       help="Number of epochs for training")
    parser.add_argument("--lr", default=1e-5, type=float,#1e-4
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
    
    #-------------------------------------------------------------------------
    # [추가] Shape Mismatch 해결 로직
    #"모양이 안 맞는 가중치는 로드하지 말고 건너뛰어라"
    #-------------------------------------------------------------------------
    model_state_dict = transformer.state_dict()
    filtered_checkpoint = {}
    
    for k, v in checkpoint.items():
        if k in model_state_dict:
            if v.shape != model_state_dict[k].shape:
                print(f"[Warning] ⚠️ Shape mismatch for '{k}': Checkpoint {v.shape} vs Model {model_state_dict[k].shape}. -> Skipped.")
                continue
        filtered_checkpoint[k] = v
        
    # # 필터링된 딕셔너리로 교체
    checkpoint = filtered_checkpoint
    # -------------------------------------------------------------------------

    #--- [수정된 부분: 예전 파라미터 살려서 로드하기] ---
    # if 'learnable_prior_patterns' in checkpoint:
    #     ckpt_param = checkpoint['learnable_prior_patterns']
    #     model_shape = transformer.learnable_prior_patterns.shape 
        
    #     if ckpt_param.dim() == 3:
    #         print(f"[Info] Reshaping 'learnable_prior_patterns' from {ckpt_param.shape} to {model_shape}")
    #         checkpoint['learnable_prior_patterns'] = ckpt_param.unsqueeze(0)
            
    #------------------------------------------------

    # #--- [수정된 부분: 예전 파라미터 살려서 로드하기] ---
    # if 'learnable_prior_patterns' in checkpoint:
    #     ckpt_param = checkpoint['learnable_prior_patterns']
    #     model_shape = transformer.learnable_prior_patterns.shape # [1, 400, 352, 352]
        
    #     # 체크포인트가 3차원 [400, 352, 352] 이라면
    #     if ckpt_param.dim() == 3:
    #         print(f"[Info] Reshaping 'learnable_prior_patterns' from {ckpt_param.shape} to {model_shape}")
            
    #         # 3차원 -> 4차원으로 변경 (맨 앞에 1 추가)
    #         # [400, 352, 352] -> [1, 400, 352, 352]
    #         checkpoint['learnable_prior_patterns'] = ckpt_param.unsqueeze(0)
            
    # #------------------------------------------------
    transformer.load_state_dict(checkpoint, strict=False)

    
    # # [추가] 새로운 Loss 함수 초기화
    criterion_vgg = VGGPerceptualLoss().cuda()
    criterion_freq = FrequencyLoss().cuda()
    # p_loss = p_loss().to("cuda")
    l2loss = torch.nn.MSELoss().to("cuda")
    class_weights = torch.tensor([1.0, 1.2]).to("cuda") 
    ce_loss = CrossEntropyLoss(weight=class_weights)
    #cafl_loss = CAFL()
    dice_loss = DiceLoss(args.num_classes)    
    # optim = torch.optim.Adam(transformer.parameters(), lr=args.lr) 기존 코드를 잠깐 주석처리함, 학습률 분리하려고 - 해결책 2
    # [설정] Dose Loss 함수 및 가중치
    criterion_dose = nn.SmoothL1Loss().cuda()
    

    # --- [해결책 2: 학습률 분리 적용 START] ---
    # 파라미터를 두 그룹(Prior Pattern과 나머지)으로 나눔
    prior_params = []
    other_params = []

    for name, param in transformer.named_parameters():
        if "learnable_prior_patterns" in name:
            prior_params.append(param)
            print(f"Detected Prior Parameter: {name}") # 확인용 출력
        else: 
            other_params.append(param)

    # Prior Pattern은 기존 LR의 1/100 (0.01배) 수준으로 아주 미세하게만 학습되도록 설정합니다.
    # 이렇게 해야 초기 물리적 구조(Radon Transform 기저)가 급격히 깨지는 것을 막을려고, 물론 추론
    optim = torch.optim.Adam([
        {'params': other_params, 'lr': args.lr},            # 나머지 네트워크: 기존 학습률 유지
        {'params': prior_params, 'lr': args.lr * 0.5}      # Prior Pattern: 1 이하의 학습률 적용
    ])
    # --- [해결책 2: 학습률 분리 적용 END] ---


    
    # --- [수정된 부분 START] ---
    # 이 두 줄을 삭제하거나 주석 처리합니다. transsino.py에서 Class Transformer 고침
    # pos_data = np.load("pos_data/prior_sino.npy")
    # pos_data = torch.from_numpy(pos_data).unsqueeze(0)
    # --- [수정된 부분 END] ---

    train_loader = AAPMDataLoader(dataset=config.aapm_dataset, batch_size=10, shuffle=True, num_workers=16, augment=False, train_val="train").get_loader()
    val_loader = AAPMDataLoader(dataset=config.aapm_dataset, batch_size=1, shuffle=False, num_workers=16, train_val="val", augment=False).get_loader()
    total_psnr = 0
    num_images = 0
    loss_history = []
    loss_val_history = []

    for epoch in range(102, args.epochs):   
        print('start epoch: ', epoch)
        epoch_loss = 0
        #t_gloss = 0
        t_l2loss = 0
        #t_fbploss = 0
        t_segloss = 0
        epoch_dose_mae = 0.0  # 누적 변수 초기화 (필수!)
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for img, full_sino, input_sino, max_value, min_value, sino_label, label, _, dose in pbar:
                # --- [디버깅: 데이터 범위 및 Shape 확인] ---
                # 학습 초반(첫 배치)에만 찍어보거나, 의심스러울 때 주석 해제
                # print(f"[DEBUG] Input Sino Range: {input_sino.min():.4f} ~ {input_sino.max():.4f}")
                # print(f"[DEBUG] Full Sino Range: {full_sino.min():.4f} ~ {full_sino.max():.4f}")
                # print(f"[DEBUG] Max Value: {max_value[0,0,0].item()}, Min Value: {min_value[0,0,0].item()}")
                # print(f"[DEBUG] Max Value Shape: {max_value.shape}") # Expected: [Batch, 1, 1] or [Batch, 1] depending on loader fix
                # -------------------------------------------
                max_value = max_value.to("cuda")
                min_value = min_value.to('cuda')
                img = img.to("cuda")
                sino = full_sino.to("cuda")
                input_sino = input_sino.to('cuda')
                # pos_data = pos_data.to(device="cuda", dtype=sino.dtype)
                sino_label = sino_label.to("cuda")
                #print("input_sino max, min: ", input_sino.max(),input_sino.min())
                #print(img.shape, sino.shape, input_sino.shape)
                label = label.to("cuda")
                # [추가] Dose 데이터 처리
                target_dose = dose.to("cuda").float()
                target_dose_log = torch.log1p(target_dose)
                #op, op2, seg_results, caf, attn_score = transformer.forward(input_sino, start_pos, pos_data,min_value, max_value)
                #op, op2, seg_results, caf, attn_score = transformer.forward(input_sino, start_pos, min_value, max_value)
                op, op2, seg_results, caf, attn_score, pred_dose = transformer.forward(input_sino, start_pos, min_value, max_value) 

                # [오류 해결] MAE 계산 시 장치 일치
                if pred_dose is not None:
                    with torch.no_grad():
                        pred_real = torch.expm1(pred_dose.view(-1))
                        pred_real = torch.clamp(pred_real, 0.0, 100.0)
                        mae = torch.abs(pred_real - target_dose.view(-1)).mean().item()
                        epoch_dose_mae += mae

                # [중요] 사이노그램 마스크(seg_results) -> 이미지 마스크(seg_iradon) 변환
                # seg_results는 [Batch, 2, 720, 768] 형태임
                # 1. 채널(Class)과 배치를 합쳐서 Radon 변환 (radon 함수가 3D/4D 입력을 받는지 확인 필요)
                #    보통 radon 함수는 [B, H, W] 또는 [B, 1, H, W]를 받으므로 reshape 필요
                B, C, H, W = seg_results.shape
                seg_reshaped = seg_results.view(B * C, H, W) 
                
                # 2. Radon 역변환 수행
                seg_img_reshaped = radon(seg_reshaped) 
                
                # 3. 다시 [Batch, Class, 300, 300]으로 복구
                new_H, new_W = seg_img_reshaped.shape[-2], seg_img_reshaped.shape[-1]
                seg_iradon = seg_img_reshaped.view(B, C, new_H, new_W)
                #seg_iradon = radon(seg_results)
                # print(seg_iradon.shape, label.shape)
                #print("seg_iradon min, max: ",seg_iradon.min(), seg_iradon.max(),"lable min, max", label.min(), label.max()) 이거는 정규화 안하는게 맞음
                loss_ce = ce_loss(seg_iradon, label[:].long())
                loss_dice = dice_loss(seg_iradon, label, softmax=True)
                seg_loss = 0.5 * loss_ce + 0.7 * loss_dice
                # print(seg_loss)

                #print("op min, max: ", op.min(), op.max())
                #print("sino min, max: ", sino.min(), sino.max())
                #print("op2 min, max: ", op2.min(), op2.max()) min=0, max=1
                # 기존 코드 주석 처리함 
                #recon_radon = op  주석 처리함 - 기존
                #loss_g = gradient_loss(op2, sino)
                loss_l2 = l2loss(op2, sino) 
                #print("op2: ", op2.amin(dim=(-2, -1), keepdim=True),op.amax(dim=(-2, -1), keepdim=True)," op: ", op.amin(dim=(-2, -1), keepdim=True),op.amax(dim=(-2, -1), keepdim=True)," sino: ", sino.amin(dim=(-2, -1), keepdim=True),sino.amax(dim=(-2, -1), keepdim=True) )
                # loss_fbp_l2 = l2loss(recon_radon, img) 주석처리함 - 기존
                # loss_fbp_gradient = gradient_loss(recon_radon, img) 주석처리함 - 기존
                # loss_abnorm = cafl_loss(caf, label)

                # --- [수정된 코드 START] ---   [해결책 1 - 보완])
                # 차원(-2, -1)은 H, W를 의미합니다. 각 이미지별로 min/max를 구합니다.
                # 1. Instance Normalization (NaN 방지)
                op_min = op.amin(dim=(-2, -1), keepdim=True)
                op_max = op.amax(dim=(-2, -1), keepdim=True)
                #print("op_min: min(), max()", op_min.min(), op_min.max(), "   op_max: mix(), max()", op_max.min(), op_max.max())
                epsilon = 1e-8
                range_op = op_max - op_min + epsilon
                
                # 분모가 0에 가까우면 recon_norm을 0으로 설정 (NaN 방지)
                recon_norm = torch.where(
                    range_op < epsilon,
                    torch.zeros_like(op),
                    (op - op_min) / range_op
                )
                #print("img: mix(), max()", img.min(), img.max())
                # 정규화된 이미지끼리 Loss를 계산합니다. (Scale Matching)
                loss_fbp_l2 = l2loss(recon_norm, img)
                #loss_fbp_gradient = gradient_loss(recon_norm, img)

                #########
                # [추가] Dose Loss 계산
                # pred_dose가 None이 아닐 때만 계산 (혹시 모를 에러 방지)

                if pred_dose is not None:
                    loss_dose_val = criterion_dose(pred_dose.view(-1), target_dose_log.view(-1))
                else:
                    loss_dose_val = 0.0

                # 3. [New] Frequency Loss (사용자 제안)
                # recon_norm(복원 이미지)과 img(정답 이미지)를 비교
                loss_low, loss_high = criterion_freq(recon_norm, img)
                loss_freq_total = loss_low + loss_high
                
                # 4. [New] VGG Perceptual Loss
                # recon_norm과 img를 비교
                loss_vgg_val = criterion_vgg(recon_norm, img)
                # --- [수정된 코드 END] ---
                #########

                # [수정 전] - 모든 항이 1.0
                #loss_t = (loss_l2 + loss_g + loss_fbp_l2 + loss_fbp_gradient) + seg_loss # + seg_loss # + loss_abnorm
                # [수정 후] - Gradient 항에 가중치를 부여합니다.
                # L2 (평균)는 1.0, Gradient (엣지)는 10.0으로 설정합니다.
                loss_t = (
                      loss_l2 * 1.0             # Sinogram L2
                    #+ loss_g * 1.1             # Sinogram Gradient (엣지 살리기)
                    + loss_fbp_l2 * 1.1         # Image L2
                    #+ loss_fbp_gradient * 1.1  # Image Gradient (선명도 복구)
                    + seg_loss * 1.0            # Segmentation Loss (가중치 5.0은 임시값, 튜닝 필요)
                    + (loss_dose_val * 0.5) # [추가]
                    + loss_freq_total * 0.8              # [New] Freq Loss (Detail & Structure)
                    + loss_vgg_val * 0.2                 # [New] Perceptual Loss
                )

                # print(loss_g, loss_l2, loss_fbp_l2, loss_fbp_gradient)
                # print(loss_l2, loss_g, loss_fbp_l2, loss_fbp_gradient)
                epoch_loss += loss_t.item()
                #t_gloss += loss_g.item()
                t_l2loss += loss_l2.item()

                # t_segloss += seg_loss.item()
                optim.zero_grad()
                loss_t.backward()
                optim.step()
                pbar.set_postfix({"Loss": loss_t.item()})
                # pbar.set_postfix({
                #     "Loss": f"{loss_t.item():.4f}", 
                #     "DoseErr": f"{current_mae:.2f}%"
                # })
                pbar.update(1)
                
        avg_epoch_loss = epoch_loss / len(train_loader)
        #avg_g_loss = t_gloss / len(train_loader)
        #avg_l2_loss = t_l2loss / len(train_loader)
        
        # avg_seg_loss = t_segloss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_epoch_loss:.4f}")
        
        save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(transformer.state_dict(), save_path)
        print(f"Model weights saved at {save_path}")
        
        # --- Validation Loop ---
        transformer.eval()
        # validation
        t_loss_val = 0
        #t_gloss_val = 0
        t_l2loss_val = 0
        #t_fbploss_val = 0
        t_dose_mae_val = 0.0 # [New] Dose 오차율 누적 변수

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
                # pos_data_val = pos_data.to('cuda') 이 줄은 원래 코드
                sino_label_val = sino_label_val.to("cuda")
                label_val = label_val.to("cuda")
                # [추가] 정답 Dose GPU 이동
                dose_val = dose_val.to("cuda").float() # [Batch] (예: 100.0, 50.0)

                # --- [수정된 부분 START] ---
                # pos_data_val = pos_data.to('cuda') # 이 줄을 삭제. 디버깅 흔적 남기기 위해 이 줄을 위의 주석과 중복해서 추가함
                # forward 호출 시 pos_data_val을 제거합니다.
                op_val, op_val2, seg_val, _, attn_score, pred_dose_val= transformer.forward(input_sino_val, start_pos, min_value_val, max_value_val)
                # --- [수정된 부분 END] ---
                # op_val, op_val2, seg_val, _, attn_score = transformer.forward(input_sino_val, start_pos, pos_data_val, min_value_val, max_value_val)

                seg_iradon = radon(seg_val)
                # print(seg_iradon.shape, label_val[:].shape)
                loss_ce = ce_loss(seg_iradon, label_val[:].long())
                loss_dice = dice_loss(seg_iradon, label_val, softmax=True)
                seg_loss_val = 0.5 * loss_ce + 0.5 * loss_dice 
                #seg_loss_val = 0.5 * loss_ce + 0.5 * loss_dice seg loss를 50배 더 크게 하기위해 주석처리함
                
                label_val = label_val.squeeze(0)
                seg_out_val = torch.argmax(torch.softmax(radon(seg_val), dim=1), dim=1).squeeze(0)

                prediction = seg_out_val.cpu().detach().numpy()
                # [디버깅용] 예측값에 1이 있는지 확인
                #print(f"Unique pred values: {np.unique(prediction)}")
                metric_i = []
                for i in range(1, vit_configs.n_classes):
                    metric_i.append(calculate_metric_percase(prediction == i, label_val.cpu().detach().numpy() == i))
                metric_list += np.array(metric_i)
                
                l2loss_val = l2loss(op_val2, sino_val)
                #gloss_val = gradient_loss(op_val2, sino_val)
                # loss_fbp_l2_val = l2loss(op_val, img_val) 해결책1,2 넣으면서 주석처리함

                # --- [Validation Loop 수정 완료] ---
                # 1. 검증 때도 똑같이 Instance Normalization 적용 (op_val 자체 통계량 사용)
                op_min_val = op_val.amin(dim=(-2, -1), keepdim=True)
                op_max_val = op_val.amax(dim=(-2, -1), keepdim=True)
                range_op_val = op_max_val - op_min_val
                epsilon = 1e-8
                
                # NaN 방지 로직 적용
                recon_norm_val = torch.where(
                    range_op_val < epsilon,
                    torch.zeros_like(op_val),
                    (op_val - op_min_val) / range_op_val
                )
                
                # 2. Loss 계산 (덮어쓰는 버그 삭제됨)
                loss_fbp_l2_val = l2loss(recon_norm_val, img_val)
                #loss_fbp_gradient_val = gradient_loss(recon_norm_val, img_val)
                # loss_fbp_gradient_val = gradient_loss(op_val, img_val) 해결책 1,2 넣으면서 주석처리함
                loss_t_val = l2loss_val + loss_fbp_l2_val + seg_loss_val # + gloss_val + loss_fbp_gradient_val
                t_l2loss_val += l2loss_val.item()
                #t_gloss_val += gloss_val.item()
                t_loss_val += loss_t_val.item()
                
                # --- [추가] 3. Dose MAE (오차율) 계산 ---
                if pred_dose_val is not None:
                    # Log Scale -> Real Scale 복원
                    pred_real = torch.expm1(pred_dose_val.view(-1))
                    pred_real = torch.clamp(pred_real, 0.0, 100.0) # 0~100% 클리핑
                    
                    # 절대 오차 (Absolute Error)
                    mae = torch.abs(pred_real - dose_val.view(-1)).mean().item()
                    t_dose_mae_val += mae

                dv_op_val = torch.clamp(op_val, 0, 1)
                
                total_psnr_val += calculate_psnr(dv_op_val, img_val)
                num_images_val += 1
            
            avg_val_loss = t_loss_val / len(val_loader)
            avg_dose_mae = t_dose_mae_val / len(val_loader) # [New] 평균 오차율
            loss_val_history.append(avg_val_loss)
        
        # [수정] 출력문에 Dose MAE 추가
        print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {avg_val_loss:.4f}, Val Dose MAE: {avg_dose_mae:.2f}%")
        #print(f"Epoch {epoch+1}/{args.epochs}, val_Loss: {avg_val_loss:.4f}")
        
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