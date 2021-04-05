# 初始化相关的技术包
import argparse
import os
import numpy as np
import math
# 初始化相关的技术包

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

# 参数加载
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  # 随机噪声z的维度
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")  # 输入图像尺寸
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 输入图像通道数
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")  # 保存生成图像和模型的间隔
opt = parser.parse_args()
print(opt)
# 参数加载

img_shape = (opt.channels, opt.img_size, opt.img_size)
print("img_shape = ", img_shape)  # 打印图像的尺寸

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):  # 定义基础的块
            layers = [nn.Linear(in_feat, out_feat)]  # 定义全连接层，两个参数代表输入和输出的维度
            if normalize:  # 决定要不要做正则化
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  # 批量归一化
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # 经过一个LeakReLU激活函数，小于0的值，斜率为0.2
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),  # 第一个block，所有的维度转成128，不使用正则化
            *block(128, 256),  # 第二个block，使用正则化，维度由128转成256
            *block(256, 512),  # 第二个block，使用正则化
            *block(512, 1024),  # 第二个block，使用正则化
            nn.Linear(1024, int(np.prod(img_shape))),  # 使用全连接层，将1024维度转化成784维
            nn.Tanh()  # 通过tanh将值域转化成[-1,1]
        )

    def forward(self, z):  # 前向传播，将输入的随机噪声z送入刚才定义的模型
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        # super(Discriminator, self)首先找到Discriminator的父类，也就是nn.Module,然后把类Discriminator的对象
        # 转换为nn.Module的对象
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 使用sigmoid，将值域变化到[0,1]
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function  损失函数
adversarial_loss = torch.nn.BCELoss()
# 损失函数使用BCE loss，Binary Cross Entropy Loss
# 是一个二项交叉熵loss，将其作为对抗loss

# Initialize generator and discriminator
generator = Generator()  # 将Generator实例化，得到一个生成器的模型实例
discriminator = Discriminator()  # 将Discriminator实例化，得到一个判别器的模型实例

# cuda acceleration
cuda = True if torch.cuda.is_available() else False
print("cuda_is_available", cuda)
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

cuda_is_available = True

# Configure data loader
# 数据格式为 28*28的灰度图，以及对应的0-9的数字标签
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(  # MNIST数据集非常常用，pytorch已经有类似的加载代码，直接调用
        "../../data/mnist",
        train=True,
        download=True,                  # 如果数据集文件夹不存在的话，就去download
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,  # 设置normalize
    shuffle=True,  # 将加载的数据打乱
)


def show_img(img, trans=True):
    # 如果是trans的话，认为img是一个tensor类型，需要将其转换为numpy类型
    if trans:
        img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))  # 转的方式是从GPU移到cpu里面，转成numpy，然后把channel维度放到最后，
        plt.imshow(img[:, :, 0], cmap="gray")
    else:
        plt.imshow(img, cmap="gray")
    plt.show()


mnist = datasets.MNIST("../../data/mnist")

for i in range(3):  # 加载minst数据集的前三张图片进行展示
    sample = mnist[i][0]
    label = mnist[i][1]
    show_img(np.array(sample), trans=False)
    print("label = ", label, "\n")

trans_resize = transforms.Resize(opt.img_size)
trans_to_tensor = transforms.ToTensor()
trans_normalize = transforms.Normalize([0.5], [0.5])  # x_n = (x-0.5)/0.5
print("shape = ", np.array(sample).shape, '\n')
print("data = ", np.array(sample), '\n')
sample = trans_resize(sample)
print("(trans_resize) shape", np.array(sample).shape, '\n')
sample = trans_to_tensor(sample)  # 转成tensor，数据从整数转成浮点数，值阈为[0,1]
print("trans_to_tensor data = ", sample, "\n")
sample = trans_normalize(sample)  # 归一化操作，值域变到[-1,1]
print("(trans_normalize) data = ", sample, '\n')

# Optimizers  使用Adam优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))      # 只优化生成器中的参数
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # 只优化判别器中的参数

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _), in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)   # 为1时，判定为真
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)    # 为1时，判定为假

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))        #从数据集中获取输入

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()             # 生成器的梯度清零

        # Sample noise as generator input    从随机向量中获取输入
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))      # 标准的正太分布，imgs.shape[0]是batchsize，
                                                                                           # opt.latent_dim参数中定义的z的维度

        # Generate a batch of images
        gen_imgs = generator(z)           # z送入生成器，得到生成图像

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)  # 将生成图像gen_imgs送入判别器，得到判别器对他的输出，然后将输出与valid计算loss，其实就是BCE

        g_loss.backward()       # 将g_loss进行反向传播
        optimizer_G.step()      # 反向传播之后再进行优化

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()      # 判别器的梯度置零

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)             # 将真实数据送入判别器，与valid计算loss
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)      # 将生成数据送入判别器，与fake计算loss，detch：梯度到gen_imgs这里不再回传
        d_loss = (real_loss + fake_loss) / 2                                      # 避免多余的计算，因为这步只想优化判别器，不优化生成器

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i       # 计算总的batch数
        if batches_done % opt.sample_interval == 0:      # 总的batch数为设定参数间隔的倍数
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)   # 保存图像

            os.makedirs("model", exist_ok=True)               # 保存模型
            torch.save(generator, 'model/generator.pkl')
            torch.save(discriminator, 'model/discriminator.pkl')
