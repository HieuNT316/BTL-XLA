import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from model import DEANet
import threading

# --- CẤU HÌNH MẶC ĐỊNH ---
DEFAULT_MODEL_PATH = '../trained_models/ITS/PSNR4131_SSIM9945.pth'

class DehazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DEA-Net: Hệ thống Khử Sương Mù Ảnh")
        self.root.geometry("1000x600")
        
        # Biến lưu trạng thái
        self.net = None
        self.input_image = None  # PIL Image gốc
        self.output_image = None # PIL Image kết quả
        self.device = 'cpu'

        # --- GIAO DIỆN ---
        
        # 1. Khung điều khiển (Trên cùng)
        control_frame = tk.Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load_model = tk.Button(control_frame, text="1. Load Model (.pth)", command=self.load_model_dialog, bg="#e1f5fe")
        self.btn_load_model.pack(side=tk.LEFT, padx=10)

        self.lbl_model_status = tk.Label(control_frame, text="Chưa load model", fg="red")
        self.lbl_model_status.pack(side=tk.LEFT, padx=5)

        self.btn_open_img = tk.Button(control_frame, text="2. Chọn Ảnh Sương Mù", command=self.open_image, state=tk.DISABLED)
        self.btn_open_img.pack(side=tk.LEFT, padx=10)

        self.btn_process = tk.Button(control_frame, text="3. XỬ LÝ (Dehaze)", command=self.process_image_thread, bg="#c8e6c9", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_process.pack(side=tk.LEFT, padx=10)
        
        self.btn_save = tk.Button(control_frame, text="4. Lưu Kết Quả", command=self.save_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=10)

        # 2. Khung hiển thị ảnh (Giữa)
        image_frame = tk.Frame(root)
        image_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Khung ảnh trái (Input)
        self.panel_left = tk.Label(image_frame, text="Ảnh Gốc (Sương mù)", bg="#eeeeee", relief="sunken")
        self.panel_left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        # Khung ảnh phải (Output)
        self.panel_right = tk.Label(image_frame, text="Kết Quả (Đã khử)", bg="#eeeeee", relief="sunken")
        self.panel_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

        # 3. Thanh trạng thái (Dưới cùng)
        self.status_bar = tk.Label(root, text="Sẵn sàng", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Tự động load model mặc định nếu có
        if os.path.exists(DEFAULT_MODEL_PATH):
            self.load_model(DEFAULT_MODEL_PATH)

    def load_model_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
        if path:
            self.load_model(path)

    def load_model(self, path):
        self.status_bar.config(text=f"Đang load model: {os.path.basename(path)}...")
        self.root.update()
        
        try:
            # Khởi tạo mạng
            self.net = DEANet(base_dim=32)
            
            # Load trọng số (Logic fix lỗi key)
            checkpoint = torch.load(path, map_location=self.device)
            if 'model' in checkpoint: state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            else: state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                # Fix lỗi conv1 -> conv1.conv1
                if ".conv1.weight" in name: name = name.replace(".conv1.weight", ".conv1.conv1.weight")
                elif ".conv1.bias" in name: name = name.replace(".conv1.bias", ".conv1.conv1.bias")
                new_state_dict[name] = v
            
            # Load với strict=False cho an toàn (dù đã fix key)
            self.net.load_state_dict(new_state_dict, strict=False)
            self.net.eval()
            self.net.to(self.device)

            self.lbl_model_status.config(text=f"Model: {os.path.basename(path)}", fg="green")
            self.btn_open_img.config(state=tk.NORMAL)
            self.status_bar.config(text="Load model thành công!")
            
        except Exception as e:
            messagebox.showerror("Lỗi Load Model", str(e))
            self.lbl_model_status.config(text="Lỗi Model!", fg="red")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.input_image = Image.open(path).convert('RGB')
            self.show_image(self.input_image, self.panel_left)
            self.btn_process.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Đã mở ảnh: {os.path.basename(path)}")
            # Xóa ảnh cũ bên phải
            self.panel_right.config(image='', text="Đang chờ xử lý...")
            self.output_image = None
            self.btn_save.config(state=tk.DISABLED)

    def show_image(self, pil_img, panel):
        # Resize ảnh để vừa khung hình hiển thị (nhưng giữ nguyên ảnh gốc trong biến)
        w, h = pil_img.size
        display_h = 400
        display_w = int(w * (display_h / h))
        
        img_resized = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        
        panel.config(image=img_tk, text="")
        panel.image = img_tk # Giữ tham chiếu để không bị garbage collection

    def process_image_thread(self):
        # Chạy trong luồng riêng để không đơ giao diện
        threading.Thread(target=self.process_image).start()

    def pad_img(self, x, patch_size=4):
        _, _, h, w = x.size()
        mod_pad_h = (patch_size - h % patch_size) % patch_size
        mod_pad_w = (patch_size - w % patch_size) % patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def process_image(self):
        if self.net is None or self.input_image is None: return

        self.status_bar.config(text="Đang xử lý... Vui lòng đợi...")
        self.btn_process.config(state=tk.DISABLED)
        
        try:
            # 1. Preprocess
            orig_w, orig_h = self.input_image.size
            img_tensor = ToTensor()(self.input_image).unsqueeze(0).to(self.device)
            img_tensor = self.pad_img(img_tensor)

            # 2. Inference
            with torch.no_grad():
                output_tensor = self.net(img_tensor)
                output_tensor = torch.clamp(output_tensor, 0, 1)

            # 3. Postprocess
            output_tensor = output_tensor[:, :, :orig_h, :orig_w] # Crop về kích thước gốc
            self.output_image = ToPILImage()(output_tensor.squeeze(0))

            # 4. Hiển thị
            self.show_image(self.output_image, self.panel_right)
            self.status_bar.config(text="Xử lý hoàn tất!")
            self.btn_save.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Lỗi Xử Lý", str(e))
            self.status_bar.config(text="Gặp lỗi khi xử lý.")
        
        self.btn_process.config(state=tk.NORMAL)

    def save_image(self):
        if self.output_image:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
            if path:
                self.output_image.save(path)
                messagebox.showinfo("Thông báo", "Đã lưu ảnh thành công!")

if __name__ == "__main__":
    root = tk.Tk()
    app = DehazeApp(root)
    root.mainloop()