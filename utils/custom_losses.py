import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

# ==========================================
# 1. VGG Perceptual Loss (수정됨: Inplace Error 해결)
# ==========================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28], use_l1=False):
        super(VGGPerceptualLoss, self).__init__()
        # VGG19 사전학습 모델 로드
        vgg = vgg19(pretrained=True).features
        self.use_l1 = use_l1
        
        # [핵심 수정] ReLU의 inplace=True를 False로 변경하여 Gradient 에러 방지
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        # 필요한 레이어만 추출하여 모듈 리스트로 만듦
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(feature_layers[0] + 1):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(feature_layers[0] + 1, feature_layers[1] + 1):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(feature_layers[1] + 1, feature_layers[2] + 1):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(feature_layers[2] + 1, feature_layers[3] + 1):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(feature_layers[3] + 1, feature_layers[4] + 1):
            self.slice5.add_module(str(x), vgg[x])
            
        # 파라미터 고정 (학습되지 않음)
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, input, target):
        # [차원 보정] 3차원 입력([B, H, W]) -> 4차원([B, 1, H, W])
        if input.dim() == 3:
            input = input.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 입력이 1채널(Grayscale)이면 3채널로 복사 (VGG는 RGB 입력 필요)
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        loss = 0.0
        x = input
        y = target
        
        # 각 슬라이스를 통과하며 특징맵 비교
        for slice_net in [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]:
            x = slice_net(x)
            y = slice_net(y)
            
            if self.use_l1:
                loss += F.l1_loss(x, y)
            else:
                loss += F.mse_loss(x, y)
                
        return loss

# ==========================================
# 2. Frequency Loss (Wavelet 기반) - 유지
# ==========================================
class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.register_buffer('haar_weights', self._get_haar_weights())

    def _get_haar_weights(self):
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        return filters

    def dwt(self, x):
        return F.conv2d(x, self.haar_weights, stride=2, padding=0, groups=1)

    def forward(self, input, target):
        # [차원 보정] 3차원 -> 4차원
        if input.dim() == 3:
            input = input.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Grayscale 처리
        if input.shape[1] > 1:
            input = input[:, 0:1, :, :]
        if target.shape[1] > 1:
            target = target[:, 0:1, :, :]

        input_dwt = self.dwt(input)   
        target_dwt = self.dwt(target) 

        input_ll = input_dwt[:, 0, :, :]
        target_ll = target_dwt[:, 0, :, :]
        
        input_hf = input_dwt[:, 1:, :, :] 
        target_hf = target_dwt[:, 1:, :, :]

        loss_low = F.l1_loss(input_ll, target_ll)
        loss_high = F.l1_loss(input_hf, target_hf)
        
        return loss_low, loss_high