import torch
import torch.nn as nn
import numpy as np
from source.dcgan_models import Generator, Discriminator
import matplotlib.pyplot as plt

class DCGAN(object):
    def __init__(self):
        self.discriminator = None
        self.generator = None
        self.criterion = None
        self.G_losses = None
        self.D_losses = None

    def build(self):
        """ Initializes Generator, Discriminator models and their weights """
        print('\nBuilding GAN model...\n')
        # Discriminator
        discriminator = Discriminator(100, 64, 3)
        discriminator.build()
        discriminator.apply(Discriminator.init_weights)
        discriminator.define_optim(0.0002, 0.5)
        # Generator
        generator = Generator(100, 64, 3)
        generator.build()
        generator.apply(Generator.init_weights)
        generator.define_optim(0.0002, 0.5)

        self.generator = generator
        self.discriminator = discriminator
        # Define Criterion
        self.criterion = nn.BCELoss()

    def print_models(self):
        """ Prints Generator and Discriminator architecture """
        if not self.discriminator or not self.generator:
            raise AttributeError('First build Discriminator and Generator.')
        else:
            print(self.discriminator)
            print('\n', '=' * 90, '\n')
            print(self.generator)

    def train(self, epochs, dataloader):
        """ Trains Discriminator and Generator """
        if not self.discriminator or not self.generator:
            raise AttributeError('First build DCGAN then train it.')
        img_list, self.G_losses, self.D_losses = [], [], []
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
                """ Train Discriminator """
                # All-real batch
                self.discriminator.zero_grad()
                real_img = data[0].to(device)
                batch_size = real_img.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                output = self.discriminator(real_img).view(-1)
                d_loss_real = self.criterion(output, label)
                d_loss_real.backward()
                d_output_real = output.mean().item()

                # All-fake batch
                noise = torch.randn(batch_size, self.discriminator.latent_vector, 1, 1, device=device)
                fake = self.generator(noise)
                label.fill_(fake_label)
                output = self.discriminator(fake.detach()).view(-1)
                d_loss_fake = self.criterion(output, label)
                d_loss_fake.backward()
                d_output_fake1 = output.mean().item()
                d_loss = d_loss_fake + d_loss_real
                self.discriminator.optimizer.step()

                """ Train Generator """
                self.generator.zero_grad()
                label.fill_(real_label)
                output = self.discriminator(fake).view(-1)
                g_loss = self.criterion(output, label)
                g_loss.backward()
                d_output_fake2 = output.mean().item()
                self.generator.optimizer.step()

                if batch_num % 50 == 0:
                    print(f'[{epoch+1}/{epochs}][{batch_num}/{len(dataloader)}]\t'
                          f'Discriminator_Loss: {round(d_loss.item(), 4)}, Generator_Loss: {round(g_loss.item(), 4)}')
                if batch_num % 100 == 0:
                    self.G_losses.append(g_loss.item())
                    self.D_losses.append(d_loss.item())
        print('\nFinished training.')

    def plot_losses(self):
        """ Plots training losses """
        print('\nPlotting losses...')
        if self.G_losses and self.D_losses:
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss")
            plt.plot(self.G_losses, label="Generator")
            plt.plot(self.D_losses, label="Discriminator")
            plt.xlabel("Iterations (every 100th)")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            raise ValueError('First train DCGAN then plot losses.')

    def generate_images(self, dataloader):
        """ Plot real and generated images with trained generator """
        print('\nGenerating images...')
        images, labels = next(iter(dataloader))
        self.generator.to('cpu')
        self.generator.eval()
        fig = plt.figure(1, figsize=(15, 5))
        gs = fig.add_gridspec(2, 6)
        for j in range(2):
            for i in range(6):
                ax = fig.add_subplot(gs[j, i], xticks=[], yticks=[])
                if j == 1:
                    noise = torch.randn(1, self.generator.latent_vector, 1, 1, device=torch.device('cpu'))
                    with torch.no_grad():
                        img = self.generator(noise)
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
        self.generator.train()
        plt.show()
