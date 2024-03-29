import torch
from torch.nn import functional
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self, path):
        self.path = path
        self.train_dataloader = None
        self.test_dataloader = None

    def build_dataset(self, image_size, batch_size):
        """ Builds DataLoader for iterating through data """
        print('\nBuilding dataset...')
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=self.path + '/train', transform=transform)
        test_dataset = datasets.ImageFolder(root=self.path + '/test', transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    @staticmethod
    def mask_images(data, option='half'):
        """ Masks the input images with option for 50%, 80% and 90% of image as pixels to reconstruct """
        valid_options = ('half', 'half_random', '10_random', '20_random', '5_random')
        multipliers = {'half': 0.5, 'half_random': 0.5, '10_random': 0.9, '20_random': 0.8, '5_random': 0.95}
        if option not in valid_options:
            raise ValueError(f"Option must be one of: {valid_options}")
        real_img = data[0]
        img_size = real_img.shape[2]
        masked_img, real_part = real_img.clone(), real_img.clone()
        masking_equations = [-1.0, -1.0, -1.0]
        if option == 'half':
            mask = np.zeros(real_img.shape[2:])
            mask[:int(img_size / 2), :] = 1
            mask = mask.astype('bool')
            for equation in masking_equations:
                masked_img[:, :, mask] = equation
                real_part[:, :, ~mask] = equation
        else:
            random_array = np.random.choice(2, int(img_size ** 2), p=[1 - multipliers[option], multipliers[option]])
            mask = random_array.reshape(real_img.shape[2:]).astype('bool')
            for equation in masking_equations:
                masked_img[:, :, mask] = equation
                real_part[:, :, ~mask] = equation
        return masked_img, real_part, mask

    @staticmethod
    def resize_images(data):
        """ Resize Tensor Images to 4 times lower resolution """
        real_img = data[0]
        resize_images = real_img.clone()
        dim_size = int(resize_images.shape[2] / 4)
        resize_images = functional.interpolate(resize_images, size=(dim_size, dim_size), mode='bilinear')
        return resize_images

    def plot_samples(self):
        """ Plots some image samples from dataloader """
        print('\nPlotting some image samples...')
        # Iterate over data with specified batch_size of 128
        images, labels = next(iter(self.train_dataloader))
        fig = plt.figure(1, figsize=(15, 5))
        for idx in range(10):
            # Make plotting grid
            ax = fig.add_subplot(2, 10 / 2, idx + 1, xticks=[], yticks=[])
            # Use image from dataloader and transform to numpy array
            img = images[idx].numpy()
            # Transpose image so that last dim is number of channels and un-normalize
            transposed_img = np.transpose(img, (1, 2, 0))
            image = transposed_img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
            plt.imshow(image)
        plt.show()
