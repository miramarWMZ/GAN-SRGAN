import torch
from torch import nn,mean
from torchvision.models.vgg import vgg19
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class generatorLoss(nn.Module):
    def __init__(self):
        super(generatorLoss,self).__init__()
        # 在论文中，生成器网络的损失函数分别是content loss 和 adversarial loss 以及 regularization loss
        vgg = vgg19(pretrained = True)
        # 加载vgg前54层，并设置为评估模式
        loss_network = nn.Sequential(*list(vgg.features)[:54]).eval()
        # 遍历这个vgg网络的全部参数，设置其参数不可被梯度下降更新
        for para in loss_network.parameters():
            para.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
    
    # out_labes out_images target_images分别是生成器生成并由鉴别器鉴别的图像的标签、生成器生成图像、目标高分辨率图像
    def forward(self,out_labels,out_images,target_images):
        # Adversarial Loss
        adversarial_loss = mean(1 - out_labels)
        # Content Loss
        content_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * content_loss + 2e-8 * tv_loss
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        # 获取批次大小
        batch_size = x.size()[0]
        # 获取图像的宽度高度
        h_x = x.size()[2]
        w_x = x.size()[3]
        # 获得图像宽度和高度上的像素总数，用于归一化
        # 从第二列到最后一列，从第二行到最后一行————避免边界效应
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        # 计算在宽度和高度方向上面的总变差异————相邻行列之间差异的平方和
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        # return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        return self.tv_loss_weight * (h_tv + w_tv) / (count_h * count_w) / batch_size   

    @staticmethod
    # 静态方法
    # 三个通道的所有像素点都要算出来
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
# 使用标签平滑，防止判别器过于自信
class SmoothLabelLoss(nn.Module):
    def __init__(self, smooth=0.1):
        super(SmoothLabelLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, output, target):
        # 平滑标签
        target = target.float()
        smooth_target = target * (1.0 - self.smooth) + (1.0 - target) * self.smooth
        
        # 使用平滑后的标签计算损失
        loss = F.binary_cross_entropy(output, smooth_target, reduction='mean')
        return loss

# ssim



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
