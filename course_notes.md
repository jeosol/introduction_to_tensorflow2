d# Notes from Tensorflow for Computer Vision Course - Full Python Tutorial for Beginners

# Youtube video link: https://www.youtube.com/watch?v=cPmjQ9V6Hbk&list=WL&index=2

# Setting up the environments

## creating virtual environment using conda

conda create -n tf2_venv python=3.9

## Activate the above environment, use:

conda activate tf2_venv

## To deactivate the environment, use:

conda deactivate

## check info (show what is the currently activated environment)

conda info --envs

# Install Tensorflow

## Requires the latest pip

pip install --upgrade pip

## Current stable release for CPU and GPU

pip install tensorflow

## Show physical devices on machine

tensorflow.config.list_physical_devices()

## Example output from above command

[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

## To check for VGA cards on your machine (Linux)

sudo lspci | grep VGA
