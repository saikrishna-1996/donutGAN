import time
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from modules import Generator, Discriminator, to_scalar, calc_gradient_penalty


kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data/fashion_mnist/', train=True, download=True,
        transform=transforms.ToTensor()
        ), batch_size=64, shuffle=False, **kwargs
    )

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data/fashion_mnist/', train=False,
        transform=transforms.ToTensor()
    ), batch_size=32, shuffle=False, **kwargs
)
test_data = list(test_loader)

netG = Generator().cuda()
netD = Discriminator().cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.cuda.FloatTensor([1])
mone = one * -1


def train(epoch):
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()

        if data.size(0) != 64:
            continue
        x_real = Variable(data, requires_grad=False).cuda()

        netD.zero_grad()
        # train with real
        D_real = netD(x_real)
        D_real = D_real.mean()
        D_real.backward(mone)

        # train with fake
        z = Variable(torch.randn(64, 128)).cuda()
        x_fake = Variable(netG(z).data)
        D_fake = netD(x_fake)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, x_real.data, x_fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

        if (batch_idx+1) % 6 == 0:
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation

            netG.zero_grad()
            z = Variable(torch.randn(64, 128)).cuda()
            x_fake = netG(z)
            D_fake = netD(x_fake)
            D_fake = D_fake.mean()
            D_fake.backward(mone)
            G_cost = -D_fake
            optimizerG.step()

            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            train_loss.append(to_scalar([D_cost, G_cost, Wasserstein_D]))
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.asarray(train_loss).mean(0),
                time.time() - start_time
            )


def test():
    z = Variable(torch.randn(64, 128)).cuda()
    x_tilde = netG(z)
    images = x_tilde.cpu().data
    save_image(images, './sample_fashion_mnist.png', nrow=8)


for i in xrange(50):
    train(i)
    test()
