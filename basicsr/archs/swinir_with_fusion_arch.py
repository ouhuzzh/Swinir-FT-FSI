import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.swinir_arch import SwinIR
from basicsr.utils.registry import ARCH_REGISTRY


# --------- SE模块 ----------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# --------- 残差密集模块 ----------
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32, num_layers=4):
        super().__init__()
        modules = []
        for i in range(num_layers):
            modules.append(nn.Conv2d(in_channels + i * growth_channels, growth_channels, 3, 1, 1))
            modules.append(nn.ReLU(inplace=True))
        self.layers = nn.ModuleList(modules)
        self.local_fusion = nn.Conv2d(in_channels + num_layers * growth_channels, in_channels, 1)

    def forward(self, x):
        features = [x]
        for i in range(0, len(self.layers), 2):
            out = self.layers[i](torch.cat(features, dim=1))
            out = self.layers[i + 1](out)
            features.append(out)
        out = self.local_fusion(torch.cat(features, dim=1))
        return out + x  # Local residual


# --------- 网络结构 ----------
@ARCH_REGISTRY.register()
class SwinIRWithFusion(SwinIR):
    def __init__(self, use_se=True, **kwargs):
        super().__init__(**kwargs)
        self.in_chans = kwargs.get('in_chans', 3)
        self.use_se = use_se

        # 多尺度特征融合层（最后4层）
        self.fuse_convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1) for _ in range(4)
        ])
        self.fuse_out = nn.Conv2d(self.embed_dim * 4, self.embed_dim, 1)

        # 注意力模块（SE or None）
        if self.use_se:
            self.attn = SELayer(self.embed_dim)
        else:
            self.attn = nn.Identity()

        # 残差密集模块
        self.rdb = ResidualDenseBlock(self.embed_dim)

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1),
            nn.Conv2d(self.embed_dim, self.in_chans, 3, 1, 1)
        )

        # 初始化所有模块
        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.fuse_convs.apply(_init)
        self.fuse_out.apply(_init)
        self.output_conv.apply(_init)
        self.rdb.apply(_init)
        if self.use_se:
            self.attn.apply(_init)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        feats = []
        for layer in self.layers:
            x = layer(x, x_size=(H, W))
            feats.append(x.permute(0, 2, 1).view(B, C, H, W))
        return feats

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = x / 255.0  # 确保输入归一化到 [0, 1]
        x_in = x

        x = self.conv_first(x)
        feats = self.forward_features(x)

        target_size = feats[-2].shape[-2:]
        fused_feats = []
        for i in range(4):
            f = self.fuse_convs[i](feats[-4 + i])
            f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            fused_feats.append(f)
        fused = torch.cat(fused_feats, dim=1)
        fused = self.fuse_out(fused)

        fused = self.attn(fused)
        fused = self.rdb(fused)

        out = self.output_conv(fused)
        return torch.clamp(out + x_in, 0, 1) * 255.0  # 恢复到 [0, 255] 范围
