#! /usr/bin/env python

from source.data_process import Data
from source.dcgan import DCGAN
from source.ae_gan import AEGAN
from source.ccgan import CCGAN
from source.baseline_cnn import CNN
from datetime import datetime
import argparse
import pandas as pd
import torch
import os
import json

parser = argparse.ArgumentParser(description='Arguments Architecture and masking options')
parser.add_argument('--arch', type=str, help='Defining which architecture to train.', 
                    choices=['DCGAN', 'AEGAN', 'CCGAN', 'CNN'], required=True)
parser.add_argument('--masking', type=str, help='Defining the masking option', 
                    choices=['half', 'half_random', '10_random', '20_random', '5_random'], default='half_random')
parser.add_argument('--epochs', type=int, help='Number of Epochs for training', default=1)
args = parser.parse_args()

data = Data('source/data/celeba')
if args.arch == 'AEGAN':
    model = AEGAN()
elif args.arch == 'CCGAN':
    model = CCGAN()
elif args.arch == 'CNN':
    model = CNN(64, 3)
else:
    model = DCGAN()
training_time = datetime.now().strftime('%Y-%m-%d-%H%M')

if __name__ == '__main__':
    save_path = f'experiments/{args.arch}/{args.masking}/run_{training_time}/generated_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    data.build_dataset(image_size=64, batch_size=128)
    data.plot_samples()
    # Build Model
    model.build()
    if args.arch == 'CNN':
        model.define_optim(learning_rate=0.001)
    model.print_models()
    start_training = input('\nStart training? [y,n] ')
    if start_training == 'y':
        # Check if architecture is DCGAN, because it is not using the masked image as input, only random noise
        if args.arch == "DCGAN":
            model.train(epochs=args.epochs, dataloader=data.train_dataloader)
            model.plot_losses()
            model.generate_images(dataloader=data.test_dataloader)
            save = input('\nSave Generator? [y,n] ')
            if save == 'y':
                torch.save(model.generator.state_dict(), f'experiments/DCGAN/{training_time}-Generator.pth')
            else:
                pass
        # Other architectures are using masked images
        else:
            model.fit(epochs=args.epochs, train_dataloader=data.train_dataloader, data_processor=data, 
                    option=args.masking, save_path=save_path, test_dataloader=data.test_dataloader)
            model.plot_losses()
            model.generate_images(test_dataloader=data.test_dataloader, data_processor=data, option=args.masking)
            model.test(test_dataloader=data.test_dataloader, data_processor=data, option=args.masking, path=save_path)
            print('\nSaving metadata... ')
            losses = model.create_losses_df()
            losses.to_csv(save_path.rsplit('/', 1)[0] + '/losses.csv')
            with open(save_path.rsplit('/', 1)[0] + '/config.json', 'w') as file:
                json.dump(model.config, file, sort_keys=True, indent=4)
            with open(save_path.rsplit('/', 1)[0] + '/metrics.json', 'w') as perf:
                json.dump(model.performance, perf, sort_keys=True, indent=4)
            save = input('\nSave Generator? [y,n] ')
            if save == 'y':
                if args.arch == 'CNN':
                    torch.save(model.state_dict(), 
                            f'experiments/{args.arch}/{args.masking}/run_{training_time}/CNN_{training_time}.pth')
                else:
                    torch.save(model.generator.state_dict(), 
                            f'experiments/{args.arch}/{args.masking}/run_{training_time}/Generator_{training_time}.pth')
            else:
                pass
    else:
        pass

