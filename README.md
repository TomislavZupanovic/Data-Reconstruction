# Data Completion and Reconstruction using Deep Learning
Using Generative Adversarial Network for sensor development based on data from various environments.

**Installation**

- Clone repository on your machine:

`$ git clone https://github.com/TomislavZupanovic/PMFST-Sensor-Augmentation.git`

- Dataset [Download](https://demo-tomislav-bucket.s3.eu-central-1.amazonaws.com/data.rar)

Unzip in cloned repository to have path like this:
 
 `.../source/data/celeba`

- Create and activate virtual environment (for Windows):

`$ python -m venv dcgan`

`$ dcgan/Scripts/activate`

- Install requirements:

`$ pip install -r requirements.txt`

- For Linux/MacOS see: [Virtual Environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

- Run training:

`$ python train.py`
