from source.inner_models import Generator, Discriminator
import torch
import numpy as np


class CGAN(object):
    def __init__(self):
        self.discriminator = None
        self.context_encoder = None
        self.C_losses = None
        self.D_losses = None

    def build(self):
        """ Initializes Context Encoder, Discriminator models and their weights """
        print('\nBuilding CGAN model...\n')
        # Dicriminator
        discriminator = Discriminator(100, 64, 3, conditional=True)
        discriminator.build()
        discriminator.apply(Discriminator.init_weights)
        discriminator.define_optim(0.0002, 0.5)
        # Context Encoder (None for latent vector size, we don't need it)
        context_encoder = Generator(None, 64, conditional=True)
        context_encoder.build()
        context_encoder.apply(Generator.init_weights)
        context_encoder.define_optim(0.0002, 0.5)

        self.context_encoder = context_encoder
        self.discriminator = discriminator

    def print_models(self):
        """ Prints Context Encoder and Discriminator architecture """
        if not self.discriminator or not self.context_encoder:
            raise AttributeError('First build Discriminator and Context Encoder.')
        else:
            print(self.discriminator)
            print('\n', '=' * 90, '\n')
            print(self.context_encoder)

    def train(self, epochs, dataloader):
        """ Trains Discriminator and Context Encoder """
        if not self.discriminator or not self.context_encoder:
            raise AttributeError('First build CGAN.')
        self.C_losses, self.D_losses = [], []
        real_label, fake_label = 1, 0
        if torch.cuda.is_available():
            print('\nGPU detected.')
            device = torch.device('cuda')
        else:
            print('\nNo GPU detected, using CPU.')
            device = torch.device('cpu')
        self.discriminator.to(device)
        self.context_encoder.to(device)
        print('\nStarting Training...')
        for epoch in range(epochs):
            for batch_num, data in enumerate(dataloader, 0):
                """ Defining mask area on images """
                real_img = data[0].to(device)
                # TODO: implement image masks

                """ Training Discriminator """
                # All-real batch
                self.discriminator.zero_grad()
                batch_size = real_img.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)