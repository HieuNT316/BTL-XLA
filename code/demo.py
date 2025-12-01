import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, ToPILImage
from PIL import Image
from model import DEANet 

# Hàm padding để kích thước ảnh chia hết cho 4 (yêu cầu của mạng U-Net)
def pad_img(x, patch_size=4):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def run_demo(args):
    # 1. Chuẩn bị thiết bị
    device = 'cpu'
    print(f"--- Đang chạy Demo trên {device.upper()} ---")

    # 2. Load Model
    print(f"Loading Model: {args.model_path}")
    net = DEANet(base_dim=32)
    
    if not os.path.exists(args.model_path):
        print(f"Lỗi: Không tìm thấy file model tại {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # --- FIX LỖI KEY (QUAN TRỌNG) ---
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        
        # Tự động map key từ file weight (deploy) sang code (training)
        if ".conv1.weight" in name:
            name = name.replace(".conv1.weight", ".conv1.conv1.weight")
        elif ".conv1.bias" in name:
            name = name.replace(".conv1.bias", ".conv1.conv1.bias")
            
        new_state_dict[name] = v
    
    try:
        net.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Warning load dict: {e}")
        print("Thử load với strict=False...")
        net.load_state_dict(new_state_dict, strict=False)

    net.eval()
    net.to(device)

    # 3. Xử lý ảnh đầu vào
    print(f"Input Image: {args.img_path}")
    if not os.path.exists(args.img_path):
        print("Lỗi: Không tìm thấy file ảnh đầu vào!")
        return

    img = Image.open(args.img_path).convert('RGB')
    orig_w, orig_h = img.size # Lưu kích thước gốc
    
    # Chuyển sang Tensor
    transform = Compose([ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Pad ảnh (tránh lỗi nếu kích thước lẻ)
    img_tensor = pad_img(img_tensor)

    # 4. Chạy mô hình
    print("Đang xử lý...")
    with torch.no_grad():
        output_tensor = net(img_tensor)
        output_tensor = torch.clamp(output_tensor, 0, 1)

    # Cắt bỏ phần padding thừa để trả về kích thước gốc
    output_tensor = output_tensor[:, :, :orig_h, :orig_w]

    # 5. Lưu kết quả
    output_img = ToPILImage()(output_tensor.squeeze(0))
    
    # Tạo tên file output (ví dụ: hanoi.jpg -> hanoi_clear.jpg)
    dir_name, file_name = os.path.split(args.img_path)
    name_only, ext = os.path.splitext(file_name)
    save_path = os.path.join(dir_name, f"{name_only}_clear{ext}")
    
    output_img.save(save_path)
    print(f"✅ Xong! Kết quả đã lưu tại: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='Đường dẫn đến ảnh sương mù')
    parser.add_argument('--model_path', type=str, default='../trained_models/ITS/PSNR4131_SSIM9945.pth', help='Đường dẫn đến file weight')
    args = parser.parse_args()
    
    run_demo(args)