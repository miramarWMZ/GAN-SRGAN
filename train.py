import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset

from utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import generatorLoss,SmoothLabelLoss,ssim
from model import generator, discriminator

# 定义脚本参数
parser = argparse.ArgumentParser(description='Train SRGAN')
parser.add_argument('--crop_size', default=96, type=int)
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8])
parser.add_argument('--train_epochs', default=100, type=int,)

if __name__ == '__main__':
    args = parser.parse_args()

    CROP_SIZE = args.crop_size
    UPSCALE_FACTOR = args.upscale_factor
    TRAIN_EPOCHS = args.train_epochs

    # 加载训练数据40%、验证数据的20%
    train_set = TrainDatasetFromFolder('data/train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_set_size = len(train_set)
    indices = list(range(train_set_size))
    train_subset_indices = indices[:train_set_size *2 // 5]
    train_subset = Subset(train_set, train_subset_indices)

    val_set = ValDatasetFromFolder('data/valid_HR', upscale_factor=UPSCALE_FACTOR)
    val_set_size = len(val_set)
    val_indices = list(range(val_set_size))
    val_subset_indices = val_indices[:val_set_size // 5]
    val_subset = Subset(val_set, val_subset_indices)

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_subset, num_workers=4, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, num_workers=4, batch_size=1, shuffle=False)

    # 初始化两个net
    netG = generator(UPSCALE_FACTOR)
    netD = discriminator()

    # 初始化生成器的损失函数
    generatorLoss = generatorLoss()

    # 使用cuda
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generatorLoss.cuda()

    # 初始化优化器，设置初始学习率和beta参数
    initial_lr = 1e-4
    later_lr = 1e-5
    optimizerG = optim.Adam(netG.parameters(), lr=initial_lr, betas=(0.9, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=initial_lr, betas=(0.9, 0.999))

    # 整体记录的所有信息
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    for epoch in range(1, TRAIN_EPOCHS + 1):
        # 动态调整学习率
        if epoch == TRAIN_EPOCHS // 2:
            for param_group in optimizerG.param_groups:
                param_group['lr'] = later_lr
            for param_group in optimizerD.param_groups:
                param_group['lr'] = later_lr

        # 一个有进度条的迭代器
        train_bar = tqdm(train_loader)
        # 每个batch中的各种信息
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        # 设置为训练模式
        netG.train()
        netD.train()
        # 防止判别器过于强大
        # 添加标签平滑
        smooth_loss = SmoothLabelLoss(smooth = 0.1)
                                              
        for data, target in train_bar:
            # 获得当前批次大小
            batch_size = data.size(0)
            # 累计批次大小，用于计算损失和得分
            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
                
            # 更新discriminator
            fake_img = netG(z)
            netD.zero_grad()
            # 计算判别器的输出
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            # 创建平滑标签
            real_target = torch.ones_like(real_out)
            fake_target = torch.zeros_like(fake_out)
            if torch.cuda.is_available():
                real_target = real_target.cuda()
                fake_target = fake_target.cuda()
            # 计算损失
            d_loss_real = smooth_loss(real_out,real_target)
            d_loss_fake = smooth_loss(fake_out,fake_target)
            d_loss = d_loss_fake + d_loss_real
            
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # 更新generator
            netG.zero_grad()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            g_loss = generatorLoss(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            # 记录当前batch的各种损失与得分
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
            # 更新进度条的描述信息
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, TRAIN_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # 迭代一个batch后，设置生成器为评估模式，同时设置结果输出路径
        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # 验证阶段
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                # 为每个epoch创建文件夹
                epoch_path = out_path + str(epoch) + '/'
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                utils.save_image(image, epoch_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
