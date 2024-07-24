from torch import nn
class residualBlock(nn.Module):
    def __init__(self,channels):
        super(residualBlock,self).__init__()
        # 第一个参数是输入通道 第二个参数是输出通道
        # k3n64s1代表卷积核大小为3*3 输出通道64 步长为1
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        # 批量归一化操作作用在channels个通道中
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        # 尽管conv2与conv1具有相同的定义，但是不要在forward中用conv1代替conv2
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self,x):
        residual = self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))
        return residual + x

# 生成器中的亚像素卷积层 用于增加图像的分辨率
# 这一个块实现了2倍的上采样    
class subpixelBlock(nn.Module):
    def __init__(self,input_channels,up_scale):
        super(subpixelBlock,self).__init__()
        # 增加特征图的通道数，准备像素重排
        self.conv = nn.Conv2d(input_channels,input_channels * up_scale ** 2,kernel_size=3,padding=1)
        # 像素重排，up_scale倍增加分辨率
        self.ps = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
    def forward(self,x):
        return self.prelu(self.ps(self.conv(x)))