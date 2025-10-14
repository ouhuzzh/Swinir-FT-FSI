import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from basicsr.utils import bgr2ycbcr


def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr."""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return (img * 255).astype(np.uint8)


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def calculate_metrics(folder1, folder2, use_y_channel=True):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    common_files = sorted(files1 & files2)

    if not common_files:
        print("没有共有的文件名，请检查文件夹内容。")
        return

    psnr_values = []
    ssim_values = []

    for filename in common_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
            continue

        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)

        img1 = read_image(path1)
        img2 = read_image(path2)

        # 确保尺寸一致
        if img1.shape != img2.shape:
            print(f"跳过 {filename}: 尺寸不一致 ({img1.shape} vs {img2.shape})")
            continue

        # 转为 HWC 格式
        img1 = reorder_image(img1)
        img2 = reorder_image(img2)

        # 使用 Y 通道
        if use_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        # 计算 PSNR
        current_psnr = psnr(img1, img2, data_range=255)
        psnr_values.append(current_psnr)

        # 计算 SSIM（如果是多通道则转为单通道）
        if img1.ndim == 3 and img1.shape[2] == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1.squeeze()
            img2_gray = img2.squeeze()

        current_ssim = ssim(img1_gray, img2_gray, data_range=255)
        ssim_values.append(current_ssim)

        print(f"{filename}: PSNR={current_psnr:.4f}, SSIM={current_ssim:.4f}")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print("\n平均指标：")
    print(f"PSNR 平均值 = {avg_psnr:.4f}")
    print(f"SSIM 平均值 = {avg_ssim:.4f}")


if __name__ == '__main__':
    folder1 = 'inputs/zzh/test/02.png'          # 替换为你的结果图文件夹
    folder2 = 'datasets/DLFSI/test_3/set12/label/02.png'     # 替换为你的真值图文件夹

    calculate_metrics(folder1, folder2, use_y_channel=True)