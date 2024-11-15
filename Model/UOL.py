import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet import Unet
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UOL(nn.Module):
    def __init__(
            self,
            sar_channels=1,
            opt_channels=3,
            hidden_dim=64,
            num_classes=1000,
            backbone='resnet50',
            decoder_channels=(256, 128, 64, 32, 16)
    ):
        super(UOL, self).__init__()
        self.Unet_SAR = Unet(encoder_name=backbone,
                             encoder_depth=5,
                             encoder_weights="imagenet",
                             classes=num_classes,
                             in_channels=sar_channels)
        self.Unet_OPT = Unet(encoder_name=backbone,
                             encoder_depth=5,
                             encoder_weights="imagenet",
                             classes=num_classes,
                             in_channels=opt_channels)
        self.Fusion_Unet_Decoder = UnetDecoder(
            encoder_channels=self.Unet_OPT.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.Fusion_Segmentation_Head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

        self.DF_layers = nn.ModuleList([
            DF(in_channels=2 * in_channels, hidden_dim=hidden_dim)
            for in_channels in self.Unet_OPT.encoder.out_channels[-5:]
        ])

    def forward(self, sar, opt):
        sar_features = self.Unet_SAR.encoder(sar)
        opt_features = self.Unet_OPT.encoder(opt)

        sar_features_cp = [x.clone().detach() for x in sar_features]
        opt_features_cp = [x.clone().detach() for x in opt_features]

        sar_decoder_output = self.Unet_SAR.decoder(*sar_features)
        opt_decoder_output = self.Unet_OPT.decoder(*opt_features)

        sar_masks = self.Unet_SAR.segmentation_head(sar_decoder_output)
        opt_masks = self.Unet_OPT.segmentation_head(opt_decoder_output)

        fusion_futures = [opt_features_cp[0] + sar_features_cp[0]]
        for i, gate_layer in enumerate(self.DF_layers):
            weight = gate_layer(opt_features_cp[i + 1], sar_features_cp[i + 1]).view(-1, 1, 1, 1)
            fusion_futures.append(weight * opt_features_cp[i + 1] + (1 - weight) * sar_features_cp[i + 1])

        fusion_decoder_output = self.Fusion_Unet_Decoder(*fusion_futures)
        fusion_masks = self.Fusion_Segmentation_Head(fusion_decoder_output)

        return fusion_masks, sar_masks, opt_masks


class DF(nn.Module):
    def __init__(self, in_channels, hidden_dim=8):
        super(DF, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, opt_features, sar_features):
        x = torch.concat([opt_features, sar_features], dim=1)
        y = self.conv(x)
        y = F.adaptive_avg_pool2d(y, 1)
        y = self.fc(y)
        return y.squeeze(-1).squeeze(-1)


if __name__ == '__main__':
    model = UOL(num_classes=6).cuda()
    print(model)

    sar_img = torch.randn(4, 1, 256, 256).cuda()
    opt_img = torch.randn(4, 3, 256, 256).cuda()

    fusion_mask, sar_mask, opt_mask = model(sar_img, opt_img)

    print(fusion_mask.shape, sar_mask.shape, opt_mask.shape)
