import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self, path):
        self.path = path
        self.dataloader = None

    def build_dataset(self, image_size, batch_size):
        """ Builds DataLoader for iterating through data """
        print('\nBuilding dataset...')
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
        dataset = datasets.ImageFolder(root=self.path, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def plot_samples(self):
        """ Plots some image samples from dataloader """
        print('\nPlotting some image samples...')
        # Iterate over data with specified batch_size of 128
        images, labels = next(iter(self.dataloader))
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
