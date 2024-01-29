import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt


def resample(data, batch_size, replace=False, return_idx=False):
    # Resample the given data sample and return the resampled data or index
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    if return_idx:
        return index
    else:
        return data[index]


class DNN_Net(nn.Module):
    def __init__(self, input_size=2, hidden_layers=2, hidden_size=100, sigma=0.02):
        super().__init__()
        self.hidden_layers = hidden_layers
        # self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers-1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size,1))
        for i in range(hidden_layers+1):
            nn.init.normal_(self.fc[i].weight, std=sigma)
            nn.init.constant_(self.fc[i].bias, 0)

    def forward(self, input):
        output = input
        for i in range(self.hidden_layers):
            output = F.elu(self.fc[i](output))
            # output = F.elu(self.batchnorm(self.fc[i](output)))
        output = self.fc[self.hidden_layers](output)
        return output


class Linear_discriminator(nn.Module):
    '''
    Fullly connect neural network for MNIST
    '''
    def __init__(self, img_shape):
        super(Linear_discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape))*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        return self.model(img)

# DV loss function for mutual information estimation
def DV_loss(D, x, y, y_ref=None, reg=False, alpha=1.0):
    '''
    D: discriminator
    x: data x
    y: data y
    '''
    # reshuffle y and concatenate with x
    joint_xy = torch.cat((x, y), dim=1)
    # if y_ref is None:
    #     marginal_xy = torch.cat((x, y[torch.randperm(x.size()[0])]), dim=1)
    # else:
    #     marginal_xy = torch.cat((x, y_ref[torch.randperm(x.size()[0])]), dim=1)
    if y_ref is None:
        x_tile = x.unsqueeze(0).repeat((y.shape[0], 1, 1, 1, 1))
        y_tile = y.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y.shape[0], -1)
    else:
        x_tile = x.unsqueeze(0).repeat((y_ref.shape[0], 1, 1, 1, 1))
        y_tile = y_ref.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y_ref.shape[0], -1)

    if reg: # add regularization term
        return -D(joint_xy.view(joint_xy.size(0), -1)).mean() + torch.logsumexp(D(marginal_xy.view(marginal_xy.size(0), -1)), dim=0).mean() - np.log(marginal_xy.shape[0]) + alpha*(torch.logsumexp(D(marginal_xy.view(marginal_xy.size(0), -1)), dim=0).mean() - np.log(marginal_xy.shape[0]))**2
    else:
        return - D(joint_xy.view(joint_xy.size(0), -1)).mean() + torch.logsumexp(D(marginal_xy.view(marginal_xy.size(0), -1)), dim=0).mean() - np.log(marginal_xy.shape[0])

# NWJ loss function for mutual information estimation
def NWJ_loss(D, x, y, y_ref=None, reg=False):
    '''
    D: discriminator
    x: data x
    y: data y
    '''

    joint_xy = torch.cat((x, y), dim=1)
    # marginal_xy = torch.cat((x, y[torch.randperm(x.size()[0])]), dim=1)
    # reshuffle y and concatenate with x
    if y_ref is None:
        x_tile = x.unsqueeze(0).repeat((y.shape[0], 1, 1, 1, 1))
        y_tile = y.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y.shape[0], -1)
    else:
        x_tile = x.unsqueeze(0).repeat((y_ref.shape[0], 1, 1, 1, 1))
        y_tile = y_ref.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
        marginal_xy = torch.cat((x_tile, y_tile), dim=-3).view(x.shape[0]*y_ref.shape[0], -1)

    return - D(joint_xy.view(joint_xy.size(0), -1)).mean() + (D(marginal_xy.view(marginal_xy.size(0), -1))).exp().mean() - 1


# InfoNCE loss function for mutual information estimation
def NCE_loss(D, x, y, y_ref=None, reg=False, alpha=1.0):
    """
    x: data x
    y: data y
    """
    if y_ref is None:
        x_tile = x.unsqueeze(0).repeat((y.shape[0], 1, 1, 1, 1))
        y_tile = y.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))
    else:
        x_tile = x.unsqueeze(0).repeat((y_ref.shape[0], 1, 1, 1, 1))
        y_tile = y_ref.unsqueeze(1).repeat((1, x.shape[0], 1, 1, 1))

    T0 = D(torch.cat([x, y], dim=1).view(x.shape[0], -1))
    if y_ref is None:
        T1 = D(torch.cat([x_tile, y_tile], dim=-3).view(x.shape[0], y.shape[0], -1))
    else:
        T1 = D(torch.cat([x_tile, y_tile], dim=-3).view(x.shape[0], y_ref.shape[0], -1))
    if reg:
        return - T0.mean() + T1.logsumexp(dim=0).mean() - np.log(T1.shape[0]) + alpha*(T1.logsumexp(dim=0) - np.log(T1.shape[0])).pow(2).mean()
    else:
        return - T0.mean() + T1.logsumexp(dim=0).mean() - np.log(T1.shape[0])

# mi estimate assuming that D approximates the optimal log density ratio
def mi_estimate(D, xy):
    return D(xy.view(xy.size(0), -1)).mean()

# PCM loss for mutual information estimation
def PCM_loss(D, x, y, y_ref=None, alpha=1.0, reg=False):
    """
    alpha: the ratio of reference samples / data samples
    """
    # make the sample size be integer
    ref_samplesize = int(x.size()[0]*alpha)
    if y_ref is None:
        xy_margin = torch.cat((x[torch.randperm(ref_samplesize)], y[torch.randperm(ref_samplesize)]), dim=1)
    else:
        xy_margin = torch.cat((x[torch.randperm(ref_samplesize)], y_ref[torch.randperm(ref_samplesize)]), dim=1)

    return - F.logsigmoid(D(torch.cat((x, y), dim=1).view(x.size()[0], -1))).mean() - alpha * F.logsigmoid(-D(xy_margin.view(ref_samplesize, -1))).mean()

class Discriminator(nn.Module):
    def __init__(self, channels=1, n_classes=10, img_size=32, code_dim=10):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code