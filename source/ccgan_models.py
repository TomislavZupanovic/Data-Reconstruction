import torch
import torch.nn as nn
import torch.optim as optim


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        model = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size, 0.8))
        model.append(nn.LeakyReLU(0.2))
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        model = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out


class Generator(nn.Module):
    """ Generator part of CCGAN """
    def __init__(self, latent_vector_size, feature_map, num_channels):
        super(Generator, self).__init__()
        self.latent_vector = latent_vector_size
        self.feature_map = feature_map
        self.channels = num_channels
        self.optimizer = None

    def build(self):
        """ U-NET network for CCGAN """
        self.down1 = UNetDown(self.channels, self.feature_map, normalize=False)
        self.down2 = UNetDown(self.feature_map, self.feature_map * 2)
        self.down3 = UNetDown((self.feature_map * 2) + self.channels, self.feature_map * 4, dropout=0.5)
        self.down4 = UNetDown(self.feature_map * 4, self.feature_map * 8, dropout=0.5)
        self.down5 = UNetDown(self.feature_map * 8, self.feature_map * 8, dropout=0.5)
        self.down6 = UNetDown(self.feature_map * 8, self.feature_map * 8, dropout=0.5)

        self.up1 = UNetUp(self.feature_map * 8, self.feature_map * 8, dropout=0.5)
        self.up2 = UNetUp(self.feature_map * 16, self.feature_map * 8, dropout=0.5)
        self.up3 = UNetUp(self.feature_map * 16, self.feature_map * 4, dropout=0.5)
        self.up4 = UNetUp(self.feature_map * 8, self.feature_map * 2)
        self.up5 = UNetUp((self.feature_map * 4) + self.channels, 64)

        final = [nn.Upsample(scale_factor=2), 
                 nn.Conv2d(self.feature_map * 2, self.channels, 3, 1, 1), 
                 nn.Tanh()]
        self.final = nn.Sequential(*final)

    def forward(self, input, x_lr):
        """ Performs forward pass """
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d2 = torch.cat((d2, x_lr), 1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)

    def define_optim(self, learning_rate, beta1):
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    @staticmethod
    def init_weights(layers):
        """ Randomly initialize weights from Normal distribution with mean = 0, std = 0.02 """
        classname = layers.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(layers.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(layers.weight.data, 1.0, 0.02)
            nn.init.constant_(layers.bias.data, 0)


class Discriminator(nn.Module):
    """ Discriminator part of CCGAN model """
    def __init__(self, latent_vector_size, feature_map, num_channels):
        super(Discriminator, self).__init__()
        self.latent_vector = latent_vector_size
        self.feature_map = feature_map
        self.channels = num_channels
        self.optimizer = None
        self.main = None

    @staticmethod
    def discriminator_block(in_filters, out_filters, stride, normalize):
        """ Returns layers of each discriminator block """
        layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def build(self):
        """ Build Discriminator model """
        layers = []
        in_filters = self.channels
        for out_filters, stride, normalize in [(self.feature_map, 2, False), (self.feature_map*2, 2, True),
                                               (self.feature_map*4, 2, True), (self.feature_map*8, 1, True)]:
            layers.extend(self.discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        """ Perform forward pass """
        return self.main(input)

    def define_optim(self, learning_rate, beta1):
        self.optimizer = optim.Adam(self.main.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    @staticmethod
    def init_weights(layers):
        """ Randomly initialize weights from Normal distribution with mean = 0, std = 0.02 """
        classname = layers.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(layers.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(layers.weight.data, 1.0, 0.02)
            nn.init.constant_(layers.bias.data, 0)
