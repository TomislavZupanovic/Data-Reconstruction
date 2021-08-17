from source.ccgan_models import Generator, Discriminator
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
import pandas as pd
import torch.nn as nn
import numpy as np


class CCGAN(object):
    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.adversarial_loss = None
        self.C_losses = None
        self.D_losses = None
        self.config = None
        
    def build(self):
        """ Initializes Generator, Discriminator models and their weights """
        print('\nBuilding CCGAN model...\n')
        latent_vector = 100,
        image_size = 64
        channels = 3
        learning_rate = 0.0002
        beta1 = 0.5
        # Discriminator
        discriminator = Discriminator(latent_vector, image_size, channels)
        discriminator.build()
        discriminator.apply(Discriminator.init_weights)
        discriminator.define_optim(learning_rate,beta1)

        # Generator(None for latent vector size, we don't need it)
        generator = Generator(None, image_size, channels)
        generator.build()
        generator.apply(Generator.init_weights)
        generator.define_optim(learning_rate,beta1)

        self.generator = generator
        self.discriminator = discriminator
        # Define Criterion
        self.adversarial_loss = nn.MSELoss()
        self.config = {
            "Channels": channels,
            "ImageSize": image_size,
            "AdversarialLoss": "MSELoss",
            "Optimizer": "Adam",
            "LearningRate": learning_rate,
            "Beta1": beta1,
            "Beta2": 0.999,
            "Epochs": None  
        }
        
    def print_models(self):
        """ Prints Generator and Discriminator architecture """
        if not self.discriminator or not self.generator:
            raise AttributeError('First build Discriminator and Generator.')
        else:
            print(self.discriminator)
            print('\n', '=' * 90, '\n')
            print(self.generator)
            
    def save_sample(self, saved_samples: dict, batches_done: int, path: str) -> None:
        # Generate inpainted image
        gen_imgs = self.generator(saved_samples["masked"], saved_samples["lowres"])
        # Save sample
        sample = torch.cat((saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data), -2)
        save_path = path + f'/{batches_done}.png'
        save_image(sample, save_path, nrow=5, normalize=True)

    def train(self, epochs, dataloader, data_processor, option, save_path):
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
        self.config['Epochs'] = epochs
        saved_samples = {}
        print('\nStarting Training...')
        for epoch in range(epochs):
            for batch_num, data in enumerate(dataloader, 0):
                real_imgs = data[0].to(device)
                masked_image, real_part, mask = data_processor.mask_images(data, option)
                resized_image = data_processor.resize_images(data)
                batch_size = real_imgs.size(0)

                # Define Real and Fake labels
                valid = torch.full((batch_size, 1, 8, 8), real_label, dtype=torch.float, device=device)
                fake = torch.full((batch_size, 1, 8, 8), fake_label, dtype=torch.float, device=device)
                
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
                    
                # Save first ten samples
                if not saved_samples:
                    saved_samples["imgs"] = real_imgs[:1].clone()
                    saved_samples["masked"] = masked_image[:1].clone()
                    saved_samples["lowres"] = resized_image[:1].clone()
                elif saved_samples["imgs"].size(0) < 10:
                    saved_samples["imgs"] = torch.cat((saved_samples["imgs"], real_imgs[:1]), 0)
                    saved_samples["masked"] = torch.cat((saved_samples["masked"], masked_image[:1]), 0)
                    saved_samples["lowres"] = torch.cat((saved_samples["lowres"], resized_image[:1]), 0)

                batches_done = epoch * len(dataloader) + batch_num
                if batches_done % 300 == 0:
                    self.save_sample(saved_samples, batches_done, save_path)
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
    
    def create_losses_df(self):
        """ Saves Generator and Discriminator Losses """
        losses = np.stack((self.D_losses, self.C_losses), axis=-1)
        losses_df = pd.DataFrame(data=losses, columns=['Discriminator', 'Generator'])
        return losses_df
        
    def generate_images(self, dataloader, data_processor, option='half'):
        """ Plot real and generated images with trained generator """
        print('\nGenerating images...')
        data = next(iter(dataloader))
        masked_images, _, _ = data_processor.mask_images(data, option)
        resized_images = data_processor.resize_images(data)
        self.generator.to('cpu')
        self.generator.eval()
        fig = plt.figure(1, figsize=(15, 5))
        gs = fig.add_gridspec(2, 6)
        for j in range(2):
            for i in range(6):
                ax = fig.add_subplot(gs[j, i], xticks=[], yticks=[])
                if j == 1: 
                    with torch.no_grad():
                        img = self.generator(torch.unsqueeze(masked_images[i], 0), 
                                             torch.unsqueeze(resized_images[i], 0))
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
        self.generator.train()
        plt.show()
