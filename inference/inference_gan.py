import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/DFSPI2/models/net_g_39000.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/ESRGAN', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23,scale=1)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        # 读取灰度图
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

        # 添加通道维度变成 (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0).float()

        # 添加 batch 维度变成 (1, 1, H, W)
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_GAN.png'), output)


if __name__ == '__main__':
    main()
