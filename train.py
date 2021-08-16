#! /usr/bin/env python

from source.data_process import Data
from source.dcgan import DCGAN
from source.ae_gan import AEGAN
from source.ccgan import CCGAN
from datetime import datetime
import argparse
import pandas as pd
import torch
import os

parser = argparse.ArgumentParser(description='Arguments Architecture and masking options')
parser.add_argument('--arch', type=str, help='Defining which architecture to train.', 
                    choices=['DCGAN', 'AEGAN', 'CCGAN'], required=True)
parser.add_argument('--masking', type=str, help='Defining the masking option', 
                    choices=['half', 'half_random', '10_random', '20_random'], required=True)
parser.add_argument('--epochs', type=int, help='Number of Epochs for training', default=1)
args = parser.parse_args()

data = Data('source/data')
if args.arch == 'AEGAN':
    model = AEGAN()
elif args.arch == 'CCGAN':
    model = CCGAN()
else:
    model = DCGAN()
training_time = datetime.now().strftime('%Y-%m-%d-%H%M')

if __name__ == '__main__':
    save_path = f'experiments/{args.arch}/{args.masking}/images_{training_time}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    data.build_dataset(image_size=64, batch_size=128)
    data.plot_samples()
    # Build Model
    model.build()
    model.print_models()
    start_training = input('\nStart training? [y,n] ')
    if start_training == 'y':
        model.train(epochs=args.epochs, dataloader=data.dataloader, data_processor=data, option=args.masking, save_path=)
        model.plot_losses()
        model.generate_images(dataloader=data.dataloader, data_processor=data, option=args.masking)
        losses = model.create_losses_df()
        losses.to_csv(save_path.rsplit('/', 1)[0] + '/losses.csv')
        save = input('\nSave Generator? [y,n] ')
        if save == 'y':
            torch.save(model.generator.state_dict(), 
                       f'experiments/{args.arch}/{args.masking}/Generator_{training_time}.pth')
        else:
            pass
    else:
        pass

