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
                real_imgs = data[0].to(device)
                masked_image, real_part, mask = data_processor.mask_images(data, option)
                resized_image = data_processor.resize_images(real_imgs)
                batch_size = real_imgs.size(0)

                # Define Real and Fake labels
                valid = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
                
                """ Train Generator"""
                self.generator.zero_grad()
                # Generate a batch of images
                generated_images = self.generator(masked_image, resized_image)
                # Loss measures generator's ability to fool the discriminator
                generator_loss = self.adversarial_loss(self.discriminator(generated_images), valid)
                generator_loss.backward()
                self.generator.optimizer.step()
                
                """ Train Discriminator """
                self.discriminator.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(generated_images.detach()), fake)
                discriminator_loss = 0.5 * (real_loss + fake_loss)
                discriminator_loss.backward()
                self.discriminator.optimizer.step()
                
                if batch_num % 50 == 0:
                    print(f'[{epoch+1}/{epochs}][{batch_num}/{len(dataloader)}]\t'
                          f'Discriminator_Loss: {round(discriminator_loss.item(), 4)}, '
                          f'Generator_Loss: {round(generator_loss.item(), 4)}')
                if batch_num % 100 == 0:
                    self.C_losses.append(generator_loss.item())
                    self.D_losses.append(discriminator_loss.item())
        print('\nFinished training.')
        
    def plot_losses(self):
        """ Plots training losses """
        print('\nPlotting losses...')
        if self.C_losses and self.D_losses:
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss")
            plt.plot(self.C_losses, label="ContextEncoder")
            plt.plot(self.D_losses, label="Discriminator")
            plt.xlabel("Iterations (every 100th)")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            raise ValueError('First train CCGAN then plot losses.')
                
    def generate_images(self, dataloader):
        """ Plot real and generated images with trained generator """
        print('\nGenerating images...')
        images, labels = next(iter(dataloader))
        self.context_encoder.to('cpu')
        self.context_encoder.eval()
        fig = plt.figure(1, figsize=(15, 5))
        gs = fig.add_gridspec(2, 6)
        for j in range(2):
            for i in range(6):
                ax = fig.add_subplot(gs[j, i], xticks=[], yticks=[])
                if j == 1:   # TODO: Change to give masked image as input instead of Latent noise
                    noise = torch.randn(1, self.context_encoder.latent_vector, 1, 1, device=torch.device('cpu'))
                    with torch.no_grad():
                        img = self.context_encoder(noise)
                    img = img.detach().cpu().numpy()
                    img = np.squeeze(img, 0)
                    title = 'Generated'
                else:
                    img = images[i].numpy()
                    title = 'Real'
                # Transpose image so that last dim is number of channels and un-normalize
                transposed_img = np.transpose(img, (1, 2, 0))
                image = transposed_img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
                ax.set_title(title)
                plt.imshow(image)
        self.context_encoder.train()
        plt.show()