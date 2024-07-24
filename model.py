from torch import nn, sigmoid, tanh
from math import log
from submodels import residualBlock, subpixelBlock

class generator(nn.Module):
    def __init__(self, scale_factor):
        # 计算放大 scale_factor 倍需要多少个上采样块
        subpixel_block_num = int(log(scale_factor, 2))
        
        super(generator, self).__init__()
        self.conv1 = nn.Sequential(
            # k9n64s1 为了保证卷积过后图像大小不变，设置填充为 4
            nn.Conv2d(3, 64, kernel_size=9, padding=4, stride=1),
            nn.PReLU()
        )
        # 论文中的图是 16 个残差块
        self.res_blocks = nn.Sequential(
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64),
            residualBlock(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        # 根据因子决定创建几个上采样块
        subpixel_blocks = [subpixelBlock(64, 2) for _ in range(subpixel_block_num)]
        subpixel_blocks.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.subpixel_blocks = nn.Sequential(*subpixel_blocks)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        res_blocks = self.res_blocks(conv1)
        conv2 = self.conv2(res_blocks)
        subpixel_blocks = self.subpixel_blocks(conv1 + conv2)
        # 将输出从 [-1, 1] 映射到 [0, 1]
        return (tanh(subpixel_blocks) + 1) / 2

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # 输入张量的形状通常都是 (batch_size, channels, height, width)，因此这里获得 batch 大小
        batch_size = x.size(0)
        # 将输出张量 self.net(x) 重塑为一个一维的张量，形状为 (batch_size,)，进而进行类别判定
        return sigmoid(self.net(x).view(batch_size))
