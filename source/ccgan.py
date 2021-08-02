from ccgan_models import Generator, Discriminator
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class CCGAN(object):
    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.adversarial_loss = None
        self.C_losses = None
        self.D_losses = None

    def build(self):
        """ Initializes Generator, Discriminator models and their weights """
        print('\nBuilding CCGAN model...\n')
        # Discriminator
        discriminator = Discriminator(100, 64, 3)
        discriminator.build()
        discriminator.apply(Discriminator.init_weights)
        discriminator.define_optim(0.0002, 0.5)

        # Generator(None for latent vector size, we don't need it)
        generator = Generator(None, 64, 3)
        generator.build()
        generator.apply(Generator.init_weights)
        generator.define_optim(0.0002, 0.5)

        self.generator = generator
        self.discriminator = discriminator
        # Define Criterion
        self.adversarial_loss = nn.MSELoss()

    def print_models(self):
        """ Prints Generator and Discriminator architecture """
        if not self.discriminator or not self.generator:
            raise AttributeError('First build Discriminator and Generator.')
        else:
            print(self.discriminator)
            print('\n', '=' * 90, '\n')
            print(self.generator)

    def train(self, epochs, dataloader, data_processor, option='half'):
        """ Trains Discriminator and Context Encoder """
        if not self.discriminator or not self.generator:
            raise AttributeError('First build AE-GAN.')
        self.C_losses, self.D_losses = [], []
        real_label, fake_label = 1, 0
        if torch.cuda.is_available():
            print('\nGPU detected.')
            device = torch.device('cuda')
        else:
            print('\nNo GPU detected, using CPU.')
            device = torch.device('cpu')
        self.discriminator.to(device)
        self.generator.to(device)
        print('\nStarting Training...')
        for epoch in range(epochs):
            for batch_num, data in enumerate(dataloader, 0):
                real_img = data[0].to(device)
                masked_image, real_part, mask = data_processor.mask_images(data, option)
                resized_image = data_processor.resize_images(real_img)
                batch_size = real_img.size(0)

                # Define Real and Fake labels
                valid = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)