import torch
import numpy as np
from source.inner_models import Generator, Discriminator
import matplotlib.pyplot as plt


class DCGAN(object):
    def __init__(self):
        self.discriminator = None
        self.generator = None
        self.G_losses = None
        self.D_losses = None

    def build(self):
        """ Initializes Generator, Discriminator models and their weights """
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

    def show_models(self):
        """ Prints Generator and Discriminator architecture """
        if not self.discriminator or not self.generator:
            raise AttributeError('First build Discriminator and Generator then show.')
        else:
            print(self.discriminator)
            print('\n', '=' * 70, '\n')
            print(self.generator)

    def train(self, epochs, dataloader):
        """ Trains Discriminator and Generator """
        if not self.discriminator or not self.generator:
            raise AttributeError('First build DCGAN then train it.')
        img_list, self.G_losses, self.D_losses = [], [], []
        real_label, fake_label = 1, 0
        if torch.cuda.is_available():
            print('GPU detected.')
            device = torch.device('cuda')
        else:
            print('No GPU detected, using CPU.')
            device = torch.device('cpu')
        self.discriminator.to(device)
        self.generator.to(device)
        print('Starting Training...')
        for epoch in range(epochs):
            for batch, data in enumerate(dataloader, 0):
                """ Train Discriminator """
                # All-real batch
                self.discriminator.zero_grad()
                real_img = data[0].to(device)
                batch_size = real_img.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                output = self.discriminator(real_img).view(-1)
                d_loss_real = self.discriminator.criterion(output, label)
                d_loss_real.backward()
                d_output_real = output.mean().item()

                # All-fake batch
                noise = torch.randn(batch_size, self.discriminator.latent_vector, 1, 1, device=device)
                fake = self.generator(noise)
                label.fill_(fake_label)
                output = self.discriminator(fake.detach()).view(-1)
                d_loss_fake = self.discriminator.criterion(output, label)
                d_loss_fake.backward()
                d_output_fake1 = output.mean().item()
                d_loss = d_loss_fake + d_loss_real
                self.discriminator.optimizer.step()

                """ Train Generator """
                self.generator.zero_grad()
                label.fill_(real_label)
                output = self.generator(fake).view(-1)
                g_loss = self.generator.criterion(output, label)
                g_loss.backward()
                d_output_fake2 = output.mean().item()
                self.generator.optimizer.step()

                if batch % 100 == 0:
                    print(f'[{epoch}/{epochs}][{batch}/{len(dataloader)}\t'
                          'D_Loss: {round(d_loss.item(), 4)}, G_Loss: {round(g_loss.item(), 4)}]')

                    self.G_losses.append(g_loss.item())
                    self.D_losses.append(d_loss.item())

    def plot_losses(self):
        """ Plots training losses """
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
        images, labels = iter(dataloader).next()
        self.generator.eval()
        fig = plt.figure(1, figsize=(12, 5))
        gs = fig.add_gridspec(2, 6)
        for j in range(2):
            for i in range(6):
                ax = fig.add_subplot(gs[j, i], xticks=[], yticks=[])
                if j == 1:
                    noise = torch.randn(1, self.generator.latent_vector, 1, 1)
                    img = self.generator(noise).detach().numpy()
                    title = 'Generated'
                else:
                    img = images[i].numpy()
                    title = 'Real'
                # Transpose image so that last dim is number of channels and un-normalize
                transposed_img = np.transpose(img, (1, 2, 0))
                image = transposed_img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
                ax.set_title(title)
                plt.imshow(image)
        plt.show()
