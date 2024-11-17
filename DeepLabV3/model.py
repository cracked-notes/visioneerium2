import torch
import torch.nn as nn
import timm


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet50",
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
        low_level_channels: int = 48,
    ):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            output_stride=16,
            out_indices=(1, 4),
        )
        aspp_in_channels = self.backbone.feature_info.channels()[-1]
        self.aspp = ASPP(in_channels=aspp_in_channels, out_channels=aspp_out_channels)
        self.decoder = Decoder(
            num_classes=num_classes,
            aspp_out_channels=aspp_out_channels,
            decoder_channels=decoder_channels,
            low_level_channels=low_level_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the backbone
        low_level_features, x = self.backbone(x)

        # Forward pass through the ASPP module
        x = self.aspp(x)

        # Forward pass through the decoder
        x = self.decoder(x, low_level_features)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        aspp_out_channels: int,
        decoder_channels: int,
        low_level_channels: int,
    ):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(aspp_out_channels, low_level_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            low_level_channels + aspp_out_channels,
            decoder_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            decoder_channels, decoder_channels, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(low_level_channels)
        self.bn2 = nn.BatchNorm2d(decoder_channels)
        self.bn3 = nn.BatchNorm2d(decoder_channels)

        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )

    def forward(
        self, x: torch.Tensor, low_level_features: torch.Tensor
    ) -> torch.Tensor:
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        low_level_features = nn.functional.interpolate(
            low_level_features, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([x, low_level_features], dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.upsample(x)

        return x


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPP, self).__init__()

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        # ASPP
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=6, dilation=6
        )
        self.conv4 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=12, dilation=12
        )
        self.conv5 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=18, dilation=18
        )

        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        image_features = self.global_avg_pool(x)
        image_features = self.conv1(image_features)
        image_features = nn.functional.interpolate(
            image_features, size=size, mode="bilinear", align_corners=False
        )

        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x3 = self.conv4(x)
        x4 = self.conv5(x)

        out = torch.cat([image_features, x1, x2, x3, x4], dim=1)
        out = self.conv6(out)

        out = nn.functional.interpolate(
            out, size=(size[0] * 4, size[1] * 4), mode="bilinear", align_corners=False
        )

        return out