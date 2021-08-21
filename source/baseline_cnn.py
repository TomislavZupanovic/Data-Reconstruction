import torch
import numpy as np
import pandas as pd
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time


class CNN(nn.Module):
    def __init__(self, feature_map: int, num_channels: int) -> None:
        super(CNN, self).__init__()
        self.feature_map = feature_map
        self.channels = num_channels
        self.optimizer = None
        self.criterion = None
        self.Losses = None
        self.config = {
            "ImageSize": feature_map,
            "Channels": num_channels,
            "LossFunction": None,
            "Optimizer": None,
            "LearningRate": None,
            "Epochs": None
        }
        self.performance = {
            "TrainingTime(s)": None,
            "AverageEpochTime(s)": None,
            "AverageBatchTime(s)": None,
            "TestMSE": None
        }
        
    def build(self) -> None:
        """ Build the CNN model """
        self.cnn_layers = nn.Sequential(
             # Input is image: channels x 64 x 64
            nn.Conv2d(self.channels, self.feature_map, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.feature_map),
            # State size: (feature_map) x 32 x 32
            nn.Conv2d(self.feature_map, self.feature_map * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.feature_map * 2),
            # State size: (feature_map * 2) x 16 x 16       
            nn.Conv2d(self.feature_map * 2, self.feature_map * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.feature_map * 4),
            # State size: (feature_map * 4) x 8 x 8
        )
        self.linear_layer = nn.Sequential(
            # State size: 16384 -> 12288 (3 x 64 x 64)
            nn.Linear(self.feature_map * 4 * 8 * 8, (self.feature_map ** 2) * self.channels)
        )
    
    def forward(self, input: Tensor) -> Tensor:
        """ Performs the forward pass through Network """
        x = self.cnn_layers(input)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x
    
    def define_optim(self, learning_rate) -> None:
        """ Initialize Loss Function and Optimizer """
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.config['LearningRate'] = learning_rate
        self.config['LossFunction'] = "MSELoss"
        self.config['Optimizer'] = 'Adam'
    
    def save_sample(self, saved_samples: dict, batches_done: int, path: str) -> None:
        """ Save some generated images to save path folder """
        # Generate inpainted image
        output = self.forward(saved_samples["masked"])
        gen_imgs = output.reshape(output.size(0), self.channels, self.feature_map, self.feature_map)
        # Save sample
        sample = torch.cat((saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data), -2)
        save_path = path + f'/{batches_done}.png'
        save_image(sample, save_path, nrow=5, normalize=True)
    
    def print_models(self):
        """ Prints architecture """
        if not self:
            raise AttributeError('First build Model.')
        else:
            print(self)
    
    @staticmethod
    def get_test_data(test_dataloader, data_processor, device, option):
        """ Returns images for testing from test dataloader """
        test_data = next(iter(test_dataloader))
        test_real_img = test_data[0].to(device)
        test_masked_image, _, _ = data_processor.mask_images(test_data, option)
        test_masked_image = test_masked_image.to(device)
        return test_real_img, test_masked_image
    
    def fit(self, epochs: int, train_dataloader, data_processor, option: str, save_path: str, test_dataloader) -> None:
        """ Trains the CNN on the training data with given number of epochs and masking option """
        self.Losses = []
        if torch.cuda.is_available():
            print('\nGPU detected.')
            device = torch.device('cuda')
        else:
            print('\nNo GPU detected, using CPU.')
            device = torch.device('cpu')
        torch.cuda.empty_cache()
        self.to(device)
        self.config['Epochs'] = epochs
        saved_samples = {}
        print('\nStarting Training...')
        training_start_time = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_num, data in enumerate(train_dataloader, 0):
                """ Defining mask area on images """
                real_img = data[0].to(device)
                masked_image, _, _ = data_processor.mask_images(data, option)
                batch_size = real_img.size(0)
                masked_image = masked_image.to(device)
                if epoch <= 1:
                    # Get testing images for generation
                    test_real_img, test_masked_image = self.get_test_data(test_dataloader, data_processor, 
                                                                     device, option)
                """ Start the training process """
                # Put gradients to zero
                self.zero_grad()
                # Perform forward pass
                output = self.forward(masked_image)
                output_image = output.reshape(batch_size, self.channels, self.feature_map, self.feature_map)
                loss = self.criterion(output_image, real_img)
                # Calculate gradients
                loss.backward()
                # Perform back propagation
                self.optimizer.step()
                running_loss += loss.item()
                """ Print the average loss """
                if batch_num % 50 == 0:
                    print(f'[{epoch+1}/{epochs}][{batch_num}/{len(train_dataloader)}], Avg. Loss: {round(running_loss / 50, 4)}')
                    running_loss = 0.0
                if batch_num % 100 == 0:
                    self.Losses.append(loss.item())
                    
                # Save first ten samples
                if not saved_samples:
                    saved_samples["imgs"] = test_real_img[:1].clone()
                    saved_samples["masked"] = test_masked_image[:1].clone()
                elif saved_samples["imgs"].size(0) < 10:
                    saved_samples["imgs"] = torch.cat((saved_samples["imgs"], test_real_img[:1]), 0)
                    saved_samples["masked"] = torch.cat((saved_samples["masked"], test_masked_image[:1]), 0)
                batches_done = epoch * len(train_dataloader) + batch_num
                if batches_done % 5000 == 0:
                    self.eval()
                    self.save_sample(saved_samples, batches_done, save_path)
                    self.train()
        total_training_time = round(time.time() - training_start_time, 0)
        self.performance['TrainingTime(s)'] = total_training_time
        self.performance['AverageEpochTime(s)'] = round(total_training_time / epochs, 1)
        self.performance['AverageBatchTime(s)'] = round((total_training_time / epochs) / len(train_dataloader), 3)
        print('\nFinished Training!')
    
    def test(self, test_dataloader, data_processor, option, path):
        """ Calculate the Test MSE and save generated test images """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.to(device)
        test_real_img, test_masked_image = self.get_test_data(test_dataloader, data_processor, 
                                                              device, option)
        batch_size = test_real_img.size(0)
        # Generate images and calculate MSE 
        output = self.forward(test_masked_image)
        output_image = output.reshape(batch_size, self.channels, self.feature_map, self.feature_map)
        test_loss = self.criterion(output_image, test_real_img)
        self.performance['TestMSE'] = round(test_loss.item(), 5)
        # Save the generated images
        sample = torch.cat((test_masked_image[:10].data, output_image[:10].data, test_real_img[:10].data), -2)
        save_path = path + f'/test.png'
        save_image(sample, save_path, nrow=5, normalize=True)
        
    def plot_losses(self):
        """ Plots training losses """
        print('\nPlotting losses...')
        if self.Losses:
            plt.figure(figsize=(10, 5))
            plt.title("Average MSELoss")
            plt.plot(self.Losses)
            plt.xlabel("Iterations (every 100th)")
            plt.ylabel("Loss")
            plt.show()
        else:
            raise ValueError('First train CNN then plot losses.')
        
    def create_losses_df(self):
        """ Saves Losses """
        losses_df = pd.DataFrame(data=self.Losses, columns=['MSELoss'])
        return losses_df
    
    def generate_images(self, test_dataloader, data_processor, option):
        """ Plot real and generated images with trained generator """
        print('\nGenerating images...')
        data = next(iter(test_dataloader))
        masked_images, _, _ = data_processor.mask_images(data, option)
        self.to('cpu')
        fig = plt.figure(1, figsize=(15, 5))
        gs = fig.add_gridspec(2, 6)
        for j in range(2):
            for i in range(6):
                ax = fig.add_subplot(gs[j, i], xticks=[], yticks=[])
                if j == 1:
                    with torch.no_grad():
                        output = self.forward(torch.unsqueeze(masked_images[i], 0))
                        img = output.reshape(1, self.channels, self.feature_map, self.feature_map)
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
        plt.show()