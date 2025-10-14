# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch


from basicsr.archs.swinir_arch import SwinIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SwinIR/xyc', help='output folder')
    parser.add_argument(
        '--task',
        type=str,
        default='classical_sr',
        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    # dn: denoising; car: compression artifact removal
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--large_model', action='store_true', help='Use large model, only used for real image sr')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/SwinIR/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)


    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # 读取图像（灰度模式）
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.  # 单通道
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).unsqueeze(0).to(device)  # 形状 (1, 1, H, W)

        # 推理和保存逻辑
        with torch.no_grad():
            # ...（填充和推理步骤不变）
            output = model(img)

        # 保存灰度图像
        output = output.squeeze().cpu().numpy()  # 形状 (H, W)
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}_SwinIR.png'), output)

        # # save image
        # output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # if output.ndim == 3:
        #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        # output = (output * 255.0).round().astype(np.uint8)
        # cv2.imwrite(os.path.join(args.output, f'{imgname}_SwinIR.png'), output)


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
    # use 'pixelshuffledirect' to save parameters
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
            # use 'nearest+conv' to avoid block artifacts
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
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
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
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')

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
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's slightly better than 1
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

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
