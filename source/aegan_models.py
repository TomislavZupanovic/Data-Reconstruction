import torch.nn as nn
import torch
import torch.optim as optim


class Generator(nn.Module):
    """ Generator part of Model """
    def __init__(self, latent_vector_size, feature_map, num_channels):
        super(Generator, self).__init__()
        self.latent_vector = latent_vector_size
        self.feature_map = feature_map
        self.channels = num_channels
        self.optimizer = None
        self.main = None

    def build(self):
        """ Builds model with Sequential definition """
        self.main = nn.Sequential(   
            # Input is image: channels x 64 x 64
            nn.Conv2d(self.channels, self.feature_map, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: feature_map x 32 x 32
            nn.Conv2d(self.feature_map, self.feature_map, 4, 2, 1),
            nn.BatchNorm2d(self.feature_map, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: feature_map x 16 x 16
            nn.Conv2d(self.feature_map, self.feature_map * 2, 4, 2, 1),
            nn.BatchNorm2d(self.feature_map * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_map * 2) x 8 x 8
            nn.Conv2d(self.feature_map * 2, self.feature_map * 4, 4, 2, 1),
            nn.BatchNorm2d(self.feature_map * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_map * 4) x 4 x 4
            nn.Conv2d(self.feature_map * 4, self.feature_map * 8, 4, 2, 1),
            nn.BatchNorm2d(self.feature_map * 8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Bottleneck
            # State size: (feature_map * 8) x 2 x 2
            nn.Conv2d(self.feature_map * 8, 4000, 1),

            # Decoder
            # State size: 4000 x 2 x 2
            nn.ConvTranspose2d(4000, self.feature_map * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.feature_map * 8, 0.8),
            nn.ReLU(),
            # state size: (feature_map*8) x 4 x 4
            nn.ConvTranspose2d(self.feature_map * 8, self.feature_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map * 4, 0.8),
            nn.ReLU(),
            # state size: (feature_map*4) x 8 x 8
            nn.ConvTranspose2d(self.feature_map * 4, self.feature_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map * 2, 0.8),
            nn.ReLU(),
            # state size: (feature_map*2) x 16 x 16
            nn.ConvTranspose2d(self.feature_map * 2, self.feature_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_map, 0.8),
            nn.ReLU(),
            # state size: (feature_map) x 32 x 32
            nn.ConvTranspose2d(self.feature_map, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output is image: channels x 64 x 64
        )
            
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


class Discriminator(nn.Module):
    """ Discriminator part of DCGAN model """
    def __init__(self, latent_vector_size, feature_map, num_channels):
        super(Discriminator, self).__init__()
        self.latent_vector = latent_vector_size
        self.feature_map = feature_map
        self.channels = num_channels
        self.optimizer = None
        self.main = None

    def build(self):
        """ Builds model with Sequential definition """
        self.main = nn.Sequential(
            # Input is image: channels x 64 x 64
            nn.Conv2d(self.channels, self.feature_map, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_map) x 32 x 32
            nn.Conv2d(self.feature_map, self.feature_map * 2, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(self.feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_map * 2) x 16 x 16
            nn.Conv2d(self.feature_map * 2, self.feature_map * 4, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(self.feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_map * 4) x 8 x 8
            nn.Conv2d(self.feature_map * 4, self.feature_map * 8, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(self.feature_map * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_map * 8) x 4 x 4
            nn.Conv2d(self.feature_map * 8, 1, 4, 1, 0, bias=True),
            # Output size: 1 x 1 x 1
        )

    def forward(self, input):
        """ Perform forward pass """
        return self.main(input)

    def define_optim(self, learning_rate, beta1):
        """ Initialize Optimizer """
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
