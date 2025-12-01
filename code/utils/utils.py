import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from PIL import Image

# --- CÁC HÀM CŨ (GIỮ NGUYÊN) ---

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def norm_zero_to_one(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def save_heat_image(x, save_path, norm=False):
    if norm:
        x = norm_zero_to_one(x)
    x = x.squeeze(dim=0)
    C, H, W = x.shape
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if C == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)[:, :, ::-1]
    x = Image.fromarray(x)
    x.save(save_path)


# --- [BỔ SUNG] CÁC HÀM TÍNH TOÁN PSNR VÀ SSIM ---

def val_psnr(pred, gt):
    # Đảm bảo giá trị trong khoảng [0, 1]
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    
    # Tính MSE
    mse = F.mse_loss(pred, gt, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(1)
    
    # Tính PSNR
    psnr = 10 * torch.log10(1 / mse)
    return psnr.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def val_ssim(img1, img2, window_size=11, size_average=True):
    # Đảm bảo input là Tensor và nằm trên cùng device (CPU/GPU)
    if not torch.is_tensor(img1):
        img1 = torch.tensor(img1)
    if not torch.is_tensor(img2):
        img2 = torch.tensor(img2)
        
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)