# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.archs.swinir_arch import SwinIR
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)


def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.patch_size,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv')

    # 002 lightweight image sr
    elif args.task == 'lightweight_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='1conv')

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv')
        else:
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=248,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='3conv')

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = SwinIR(
            upscale=1,
            in_chans=1,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=4,
            upsampler='',
            resi_connection='3conv')

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')

    # 006 JPEG compression artifact reduction
    elif args.task == 'jpeg_car':
        model = SwinIR(
            upscale=1,
            in_chans=1,
            img_size=126,
            window_size=7,
            img_range=255.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')

    else:
        raise ValueError(f"Unsupported task: {args.task}")

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SwinIR/xyc', help='output folder')
    parser.add_argument('--task', type=str, default='gray_dn',
                        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--large_model', action='store_true', help='Use large model, only used for real image sr')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')

    args = parser.parse_args(args)  # 如果是 GUI 调用，则 args 是一个列表；否则从 sys.argv 获取

    logging.info(f"开始推理任务，输入: {args.input}, 输出: {args.output}, 模型: {args.model_path}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # 遍历输入文件夹中的所有图像
    input_files = sorted(glob.glob(os.path.join(args.input, '*')))
    if not input_files:
        logging.warning("输入文件夹为空！")
        return

    for idx, path in enumerate(input_files):
        if not os.path.isfile(path):
            continue
        imgname = os.path.splitext(os.path.basename(path))[0]
        logging.info(f'Processing {idx + 1}/{len(input_files)}: {imgname}')

        try:
            # 读取图像（灰度模式）
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # 单通道
            img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)  # 形状 (1, 1, H, W)

            with torch.no_grad():
                output = model(img)

            # 后处理并保存结果
            output = output.squeeze().cpu().numpy()
            output = (output * 255.0).round().astype(np.uint8)
            output_path = os.path.join(args.output, f'{imgname}_SwinIR.png')
            cv2.imwrite(output_path, output)
            logging.info(f'Saved result to {output_path}')
        except Exception as e:
            logging.error(f"处理失败: {path} - 错误: {e}")

    logging.info("推理完成！")


if __name__ == '__main__':
    main()