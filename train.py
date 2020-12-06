from source.data_process import Data
from source.dcgan import DCGAN
from datetime import datetime
import torch

data = Data('source/data')
dcgan = DCGAN()
model_train_time = datetime.now().strftime('%Y%m%d-%H%M')

if __name__ == '__main__':
    data.build_dataset(image_size=64, batch_size=128)
    data.plot_samples()
    # Build DCGAN
    dcgan.build()
    dcgan.show_models()
    start_training = input('\nStart training? [y,n] ')
    if start_training == 'y':
        dcgan.train(epochs=1, dataloader=data.dataloader)
        dcgan.plot_losses()
        dcgan.generate_images(dataloader=data.dataloader)
        save = input('\nSave Generator? [y,n] ')
        if save == 'y':
            torch.save(dcgan.generator.state_dict(), f'saved_models/{model_train_time}-Generator.pth')
        else:
            pass
    else:
        pass

