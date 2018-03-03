import torch
import torch.nn as nn
from torch.autograd import Variable


def to_scalar(arr):
    if type(arr) == list:
        return [x.cpu().data.tolist()[0] for x in arr]
    else:
        return arr.cpu().data.tolist()[0]


def calc_gradient_penalty(netD, real_data, fake_data, lamda=20):
    bsz = real_data.size(0)
    alpha = torch.rand(bsz, 1, 1, 1)
    alpha = alpha.expand(real_data.size()).cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda
    return gradient_penalty


class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * dim),
            nn.BatchNorm1d(4 * 4 * 4 * dim),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 5),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 5),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )
        self.deconv_out = nn.Sequential(
            nn.ConvTranspose2d(dim, 1, 8, stride=2),
            nn.Sigmoid()
        )
        self.dim = dim

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.output = nn.Linear(4 * 4 * 4 * dim, 1)
        self.dim = dim

    def forward(self, input):
        # input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * self.dim)
        out = self.output(out)
        return out.view(-1)


if __name__ == '__main__':
    netG = Generator().cuda()
    netD = Discriminator().cuda()

    real_x = Variable(torch.randn(32, 1, 28, 28).cuda(), requires_grad=False)
    z = Variable(torch.randn(32, 128), requires_grad=False).cuda()
    fake_x = netG(z)
    print fake_x.size()

    fake_logit = netD(fake_x)
    real_logit = netD(real_x)

    print fake_logit.size()
    print real_logit.size()
