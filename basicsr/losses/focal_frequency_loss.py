import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss.

    Reference:
        Focal Frequency Loss for Image Reconstruction and Synthesis. ICCV 2021.
        https://arxiv.org/pdf/2012.12821.pdf

    Args:
        loss_weight (float): Overall weight for the loss. Default: 1.0.
        alpha (float): Exponent scaling for the dynamic weight matrix. Default: 1.0.
        patch_factor (int): Divide image into patch_factor × patch_factor patches. Default: 1.
        ave_spectrum (bool): Whether to average spectra over batch. Default: False.
        log_matrix (bool): Whether to use log for weight matrix. Default: False.
        batch_matrix (bool): Whether to normalize matrix globally over batch. Default: False.
    """

    def __init__(self,
                 loss_weight=1.0,
                 alpha=1.0,
                 patch_factor=1,
                 ave_spectrum=False,
                 log_matrix=False,
                 batch_matrix=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

        # check PyTorch version
        version_nums = tuple(map(int, torch.__version__.split('+')[0].split('.')))
        self.is_high_version = version_nums > (1, 7, 1)

    def tensor2freq(self, x):
        """Convert spatial image to frequency domain (FFT2)."""
        pf = self.patch_factor
        N, C, H, W = x.shape
        if H % pf != 0 or W % pf != 0:
            raise ValueError(f'Image size ({H}, {W}) not divisible by patch_factor {pf}')
        ph, pw = H // pf, W // pf

        # split into patches
        patches = [
            x[:, :, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            for i in range(pf) for j in range(pf)
        ]
        patches = torch.stack(patches, dim=1)

        # FFT
        if self.is_high_version:
            freq = torch.fft.fft2(patches, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], dim=-1)
        else:
            freq = torch.rfft(patches, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        """Compute weighted frequency domain loss."""
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            diff = (recon_freq - real_freq) ** 2
            mag = torch.sqrt(diff[..., 0] + diff[..., 1]) ** self.alpha

            if self.log_matrix:
                mag = torch.log(mag + 1.0)

            if self.batch_matrix:
                mag = mag / mag.max()
            else:
                mag = mag / mag.amax(dim=(-1, -2), keepdim=True).amax(dim=1, keepdim=True)

            mag = torch.nan_to_num(mag, nan=0.0, posinf=1.0, neginf=0.0)
            mag = mag.clamp(0.0, 1.0)
            weight_matrix = mag.detach()

        diff = (recon_freq - real_freq) ** 2
        freq_distance = diff[..., 0] + diff[..., 1]
        loss = weight_matrix * freq_distance
        return loss.mean()

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward pass.

        Args:
            pred (Tensor): (N, C, H, W)
            target (Tensor): (N, C, H, W)
            matrix (Tensor, optional): Predefined weighting matrix.
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = pred_freq.mean(dim=0, keepdim=True)
            target_freq = target_freq.mean(dim=0, keepdim=True)

        return self.loss_weight * self.loss_formulation(pred_freq, target_freq, matrix)
