import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- IMPORT MODULE ---
from utils.utils import val_psnr, val_ssim
from data.data_loader import TestDataset
from model import DEANet  

def eval(val_loader, network):
    network.eval()
    psnr_list = []
    ssim_list = []

    for batch in tqdm(val_loader, desc='evaluation'):
        inputs, targets, name = batch
        
        # Ép chạy CPU
        inputs = inputs.cpu()
        targets = targets.cpu()

        with torch.no_grad():
            pred = network(inputs)
            pred = torch.clamp(pred, 0, 1)

        psnr_val = val_psnr(pred, targets)
        ssim_val = val_ssim(pred, targets).item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    return avg_psnr, avg_ssim

if __name__ == '__main__':
    # --- 1. KHAI BÁO THAM SỐ ---
    parser = argparse.ArgumentParser()
    # Thêm tham số subset để chọn indoor hay outdoor
    parser.add_argument('--dataset', default='SOTS', type=str)
    parser.add_argument('--subset', default='outdoor', type=str, help='indoor hoặc outdoor (chỉ dùng cho SOTS)')
    parser.add_argument('--model_name', default='DEA-Net-CR', type=str)
    parser.add_argument('--pre_trained_model', default='PSNR4131_SSIM9945.pth', type=str)
    args = parser.parse_args()

    # --- 2. CẤU HÌNH ĐƯỜNG DẪN (PATH) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    dataset_root = os.path.join(project_root, 'dataset')

    print(f"Dataset Root: {dataset_root}")

    # [QUAN TRỌNG] Xử lý riêng cho SOTS
    if args.dataset == 'SOTS':
        # Cấu trúc: dataset/SOTS/outdoor/hazy (Không có folder test)
        # Lấy subset từ tham số (mặc định là outdoor)
        subset = args.subset 
        
        test_hazy_path = os.path.join(dataset_root, 'SOTS', subset, 'hazy')
        test_clear_path = os.path.join(dataset_root, 'SOTS', subset, 'clear')
        
        print(f"--- Đang chạy SOTS (Chế độ: {subset}) ---")
        
    elif args.dataset == 'ITS':
        # Cấu trúc: dataset/ITS/test/hazy (Hoặc dataset/ITS/train/hazy)
        if os.path.exists(os.path.join(dataset_root, 'ITS', 'test')):
            base_dir = os.path.join(dataset_root, 'ITS', 'test')
        else:
            print("Warning: Không thấy ITS/test, dùng ITS/train để demo.")
            base_dir = os.path.join(dataset_root, 'ITS', 'train')
            
        test_hazy_path = os.path.join(base_dir, 'hazy')
        test_clear_path = os.path.join(base_dir, 'clear')
        print("--- Đang chạy ITS ---")

    else:
        print(f"Lỗi: Dataset '{args.dataset}' chưa hỗ trợ.")
        exit(1)

    # Kiểm tra đường dẫn
    if not os.path.exists(test_hazy_path):
        print(f"LỖI: Không tìm thấy folder ảnh tại: {test_hazy_path}")
        print(f"Bạn đang chọn subset: '{args.subset}'. Hãy kiểm tra xem folder '{args.subset}' có tồn tại trong SOTS không.")
        exit(1)

    print(f"Input path: {test_hazy_path}")

    # --- 3. CHẠY CODE ---
    val_dataset = TestDataset(test_hazy_path, test_clear_path)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

    network = DEANet(base_dim=32)
    
    # Load Model
    model_path = args.pre_trained_model
    if not os.path.exists(model_path):
         # Tìm trong folder trained_models
        potential_path = os.path.join(project_root, 'trained_models', args.pre_trained_model)
        if os.path.exists(potential_path):
            model_path = potential_path
        else:
             print(f"Lỗi: Không tìm thấy file model {model_path}")
             exit(1)
        
    # ... đoạn trên giữ nguyên ...
    
    print(f"Loading model form: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # ... (đoạn trên giữ nguyên) ...
    
    # Loại bỏ prefix 'module.' nếu có
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        
        # --- FIX LỖI 2 LẦN CHỮ CONV1 ---
        # Weight file: "block.conv1.weight"
        # Code hiện tại: "block.conv1.conv1.weight"
        # -> Ta cần đổi tên key trong file weight từ "conv1" thành "conv1.conv1" để khớp với code
        
        # Kiểm tra nếu key kết thúc bằng ".conv1.weight" hoặc ".conv1.bias"
        if ".conv1.weight" in name:
            name = name.replace(".conv1.weight", ".conv1.conv1.weight")
        elif ".conv1.bias" in name:
            name = name.replace(".conv1.bias", ".conv1.conv1.bias")
            
        new_state_dict[name] = v
            
    # Load model
    print("Đang load model với key mapping fix...")
    network.load_state_dict(new_state_dict)
    
    # ... (đoạn dưới giữ nguyên) ... 
    
    network = network.cpu()
    
    # ... đoạn dưới giữ nguyên ...

    print("Start Evaluation...")
    avg_psnr, avg_ssim = eval(val_loader, network)
    print(f'\n[KẾT QUẢ] Dataset: {args.dataset}-{args.subset} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}')