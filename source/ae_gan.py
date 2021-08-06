from source.inner_models import Generator, Discriminator
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class AEGAN(object):
    def __init__(self):
        self.discriminator = None
        self.context_encoder = None
        self.C_losses = None
        self.D_losses = None
        self.adversarial_loss = None
        self.pixelwise_loss = None

    def build(self):
        """ Initializes Context Encoder, Discriminator models and their weights """
        print('\nBuilding AE-GAN model...\n')
        # Discriminator
        discriminator = Discriminator(100, 64, 3, gan_option='AE-GAN')
        discriminator.build()
        discriminator.apply(Discriminator.init_weights)
        discriminator.define_optim(0.0002, 0.5)
        # Context Encoder (None for latent vector size, we don't need it)
        context_encoder = Generator(None, 64, 3, gan_option='AE-GAN')
        context_encoder.build()
        context_encoder.apply(Generator.init_weights)
        context_encoder.define_optim(0.0002, 0.5)

        self.context_encoder = context_encoder
        self.discriminator = discriminator
        # Define Criterion
        self.adversarial_loss = nn.MSELoss()
        self.pixelwise_loss = nn.L1Loss()

    def print_models(self):
        """ Prints Context Encoder and Discriminator architecture """
        if not self.discriminator or not self.context_encoder:
            raise AttributeError('First build Discriminator and Context Encoder.')
        else:
            print(self.discriminator)
            print('\n', '=' * 90, '\n')
            print(self.context_encoder)

    def train(self, epochs, dataloader, data_processor, option='half'):
        """ Trains Discriminator and Context Encoder """
        if not self.discriminator or not self.context_encoder:
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
        self.context_encoder.to(device)
        print('\nStarting Training...')
        for epoch in range(epochs):
            for batch_num, data in enumerate(dataloader, 0):
                """ Defining mask area on images """
                real_img = data[0].to(device)
                masked_image, real_part, mask = data_processor.mask_images(data, option)
                batch_size = real_img.size(0)

                # Define Real and Fake labels
                valid = torch.full((batch_size, 1, 1, 1), real_label, dtype=torch.float, device=device)
                fake = torch.full((batch_size, 1, 1, 1), fake_label, dtype=torch.float, device=device)

                """ Training Context Encoder """
                # All-real batch
                self.context_encoder.zero_grad()
                # Generate a batch of images
                generated_parts = self.context_encoder(masked_image)
                # Adversarial and pixelwise loss
                g_adv = self.adversarial_loss(self.discriminator(generated_parts), valid)
                g_pixel = self.pixelwise_loss(generated_parts, real_img)
                # Total loss
                context_enc_loss = 0.001 * g_adv + 0.999 * g_pixel
                context_enc_loss.backward()
                self.context_encoder.optimizer.step()

                """ Training Discriminator """
                self.discriminator.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_img), valid)
                fake_loss = self.adversarial_loss(self.discriminator(generated_parts.detach()), fake)
                discriminator_loss = 0.5 * (real_loss + fake_loss)
                discriminator_loss.backward()
                self.discriminator.optimizer.step()

                if batch_num % 50 == 0:
                    print(f'[{epoch+1}/{epochs}][{batch_num}/{len(dataloader)}]\t'
                          f'Discriminator_Loss: {round(discriminator_loss.item(), 4)}, '
                          f'Context_Loss: {round(context_enc_loss.item(), 4)}')
                if batch_num % 100 == 0:
                    self.C_losses.append(context_enc_loss.item())
                    self.D_losses.append(discriminator_loss.item())
        print('\nFinished training.')

    def plot_losses(self):
        """ Plots training losses """
        print('\nPlotting losses...')
        if self.C_losses and self.D_losses:
            plt.figure(figsize=(10, 5))
            plt.title("ContextEncoder and Discriminator Loss")
            plt.plot(self.C_losses, label="ContextEncoder")
            plt.plot(self.D_losses, label="Discriminator")
            plt.xlabel("Iterations (every 100th)")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            raise ValueError('First train AE-GAN then plot losses.')

    def generate_images(self, dataloader, data_processor, option='half'):
        """ Plot real and generated images with trained generator """
        print('\nGenerating images...')
        data = next(iter(dataloader))
        masked_images, _, _ = data_processor.mask_images(data, option)
        self.context_encoder.to('cpu')
        self.context_encoder.eval()
        fig = plt.figure(1, figsize=(15, 5))
        gs = fig.add_gridspec(2, 6)
        for j in range(2):
            for i in range(6):
                ax = fig.add_subplot(gs[j, i], xticks=[], yticks=[])
                if j == 1:
                    with torch.no_grad():
                        img = self.context_encoder(torch.unsqueeze(masked_images[i], 0))
                    img = img.detach().cpu().numpy()
                    img = np.squeeze(img, 0)
                    title = 'Reconstructed'
                else:
                    img = masked_images[i].numpy()
                    title = 'Masked'
                # Transpose image so that last dim is number of channels and un-normalize
                transposed_img = np.transpose(img, (1, 2, 0))
                image = transposed_img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
                ax.set_title(title)
                plt.imshow(image)
        self.context_encoder.train() 
        plt.show()
