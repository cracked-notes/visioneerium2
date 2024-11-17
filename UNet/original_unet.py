import torch
import torch.nn as nn


def double_conv_block(in_c: int, out_c: int) -> nn.Sequential:
    """
    Double convolution block with ReLU activation.

    Args:
        in_c: number of input channels.
        out_c: number of output channels.

    Returns:
        Sequential: Sequential layer containing two convolution layers with ReLU activation.
    """
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv


class UNet(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = double_conv_block(in_channels, 64)
        self.down_conv_2 = double_conv_block(64, 128)
        self.down_conv_3 = double_conv_block(128, 256)
        self.down_conv_4 = double_conv_block(256, 512)
        self.down_conv_5 = double_conv_block(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv_block(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv_block(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv_block(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv_block(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Encoder
        ec1 = self.down_conv_1(image)
        em1 = self.max_pool_2x2(ec1)

        ec2 = self.down_conv_2(em1)
        em2 = self.max_pool_2x2(ec2)

        ec3 = self.down_conv_3(em2)
        em3 = self.max_pool_2x2(ec3)

        ec4 = self.down_conv_4(em3)
        em4 = self.max_pool_2x2(ec4)

        ec5 = self.down_conv_5(em4)

        # Decoder
        d = self.up_trans_1(ec5)
        dc = ec4
        d = self.up_conv_1(torch.cat([d, dc], 1))

        d = self.up_trans_2(d)
        dc = ec3
        d = self.up_conv_2(torch.cat([d, dc], 1))

        d = self.up_trans_3(d)
        dc = ec2
        d = self.up_conv_3(torch.cat([d, dc], 1))

        d = self.up_trans_4(d)
        dc = ec1
        d = self.up_conv_4(torch.cat([d, dc], 1))

        d = self.out(d)
        return d


# Debug testing code
if __name__ == "__main__":
    image = torch.rand((6, 3, 256, 256))  # Batch of 6 RGB images
    model = UNet()
    output = model(image)
    print(f"Output shape: {output.shape}")
